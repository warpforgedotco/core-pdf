from __future__ import annotations

import pytest

from core_pdf.impl.exceptions import PdfDecryptionError, PdfParseError, PdfUnsupportedError
from core_pdf.impl.spec.s_07_security.ciphers import (
    internal_aes_cbc_decrypt,
    internal_aes_cbc_encrypt,
    internal_aes_ecb_decrypt,
    internal_aes_gcm_decrypt,
    internal_rc4_crypt,
)
from core_pdf.impl.spec.s_07_security.standard import (
    create_standard_decipher,
    internal_CryptMethod,
    internal_normalize_password,
    internal_parse_config,
    internal_parse_crypt_filters,
    internal_resolve_crypt_method,
    internal_saslprep,
    internal_StandardSecurityConfig,
    internal_StandardSecurityHandler,
    internal_stream_crypt_filter_name,
)
from core_pdf.impl.spec.s_07_syntax.types import PdfDict
from core_pdf.impl.types import PdfName
from tests.helpers.paths import FIXTURES, require_fixture


def test_aes_cbc_known_vector() -> None:
    key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    initialization_vector = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
    plaintext = bytes.fromhex("6bc1bee22e409f96e93d7e117393172a")
    expected = bytes.fromhex("7649abac8119b246cee98e9b12e9197d")

    encrypted = internal_aes_cbc_encrypt(
        key,
        initialization_vector,
        plaintext,
        use_padding=False,
    )

    assert encrypted == expected
    assert (
        internal_aes_cbc_decrypt(
            key,
            initialization_vector,
            encrypted,
            use_padding=False,
        )
        == plaintext
    )


def test_aes_cbc_pkcs7_padding_round_trip() -> None:
    key = bytes(32)
    initialization_vector = bytes(range(16))
    plaintext = b"not aligned to an AES block"
    encrypted = internal_aes_cbc_encrypt(
        key,
        initialization_vector,
        plaintext,
        use_padding=True,
    )

    assert (
        internal_aes_cbc_decrypt(
            key,
            initialization_vector,
            encrypted,
            use_padding=True,
        )
        == plaintext
    )


@pytest.mark.parametrize(
    ("initialization_vector", "ciphertext"),
    [
        (bytes(15), bytes(16)),
        (bytes(16), bytes(15)),
    ],
)
def test_aes_cbc_rejects_invalid_block_shape(
    initialization_vector: bytes,
    ciphertext: bytes,
) -> None:
    with pytest.raises(PdfDecryptionError, match="Invalid encrypted object ciphertext"):
        internal_aes_cbc_decrypt(
            bytes(16),
            initialization_vector,
            ciphertext,
            use_padding=False,
        )


def test_aes_ecb_decrypts_known_vector() -> None:
    key = bytes.fromhex("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4")
    ciphertext = bytes.fromhex("f3eed1bdb5d2a03c064b5a7e3db181f8")
    expected = bytes.fromhex("6bc1bee22e409f96e93d7e117393172a")

    assert internal_aes_ecb_decrypt(key, ciphertext) == expected


def test_aes_gcm_decrypts_nist_aes_256_known_vector() -> None:
    """NIST CAVP 14.0 AES-256-GCM, PTlen=128/AADlen=0, Count 0."""
    key = bytes.fromhex("31bdadd96698c204aa9ce1448ea94ae1fb4a9a0b3c9d773b51bb1822666b8f22")
    initialization_vector = bytes.fromhex("0d18e06c7c725ac9e362e1ce")
    plaintext = bytes.fromhex("2db5168e932556f8089a0622981d017d")
    ciphertext = bytes.fromhex("fa4362189661d163fcd6a56d8bf0405a")
    authentication_tag = bytes.fromhex("d636ac1bbedd5cc3ee727dc2ab4a9489")

    # ISO/TS 32003:2023, 5.2 serializes AESV4 objects as IV, ciphertext,
    # then tag and requires nil additional authenticated data.
    serialized = initialization_vector + ciphertext + authentication_tag

    assert internal_aes_gcm_decrypt(key, serialized) == plaintext


@pytest.mark.parametrize("byte_index", [0, 12, -1], ids=["iv", "ciphertext", "tag"])
def test_aes_gcm_authenticates_every_serialized_component(byte_index: int) -> None:
    data = bytearray(
        bytes.fromhex("0d18e06c7c725ac9e362e1ce")
        + bytes.fromhex("fa4362189661d163fcd6a56d8bf0405a")
        + bytes.fromhex("d636ac1bbedd5cc3ee727dc2ab4a9489")
    )
    data[byte_index] ^= 1

    with pytest.raises(PdfDecryptionError, match="Invalid encrypted object ciphertext"):
        internal_aes_gcm_decrypt(
            bytes.fromhex("31bdadd96698c204aa9ce1448ea94ae1fb4a9a0b3c9d773b51bb1822666b8f22"),
            bytes(data),
        )


def test_aes_gcm_rejects_truncated_serialized_object() -> None:
    # ISO/TS 32003:2023, 5.2 requires a 12-byte IV and 16-byte tag even
    # when the encrypted plaintext is empty.
    with pytest.raises(PdfDecryptionError, match="Invalid encrypted object ciphertext"):
        internal_aes_gcm_decrypt(bytes(32), bytes(27))


def test_aes_gcm_requires_the_iso_ts_32003_key_size() -> None:
    with pytest.raises(ValueError, match="AESV4 key must be 32 bytes"):
        internal_aes_gcm_decrypt(bytes(16), bytes(28))


def test_rc4_known_vector() -> None:
    key = bytes.fromhex("0102030405")
    plaintext = bytes(16)
    encrypted = internal_rc4_crypt(key, plaintext)

    assert encrypted.hex() == "b2396305f03dc027ccc3524a0a1118a8"
    assert internal_rc4_crypt(key, encrypted) == plaintext


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({}, "Invalid encryption dictionary"),
        ({"Filter": PdfName.of("PubSec")}, "Public-key encryption"),
        ({"Filter": PdfName.of("Custom")}, "Unsupported encryption filter"),
        (
            {"Filter": PdfName.of("Standard"), "V": 99},
            "Unsupported standard encryption algorithm",
        ),
        (
            {"Filter": PdfName.of("Standard"), "V": 3},
            "Unsupported standard encryption algorithm",
        ),
        (
            {"Filter": PdfName.of("Standard"), "V": 7},
            "Unsupported standard encryption algorithm",
        ),
    ],
)
def test_standard_security_factory_rejects_unsupported_handler(
    params: PdfDict,
    message: str,
) -> None:
    with pytest.raises(PdfUnsupportedError, match=message):
        create_standard_decipher([b"document-id"], params)


@pytest.mark.parametrize(
    ("version", "revision"),
    [
        (1, 4),
        (2, 2),
        (4, 3),
        (5, 4),
        (6, 6),
    ],
)
def test_standard_security_factory_rejects_mismatched_revision(
    version: int,
    revision: int,
) -> None:
    params: PdfDict = {
        "Filter": PdfName.of("Standard"),
        "V": version,
        "R": revision,
        "P": -4,
        "O": bytes(32),
        "U": bytes(32),
    }

    with pytest.raises(PdfUnsupportedError, match="Invalid encryption dictionary"):
        create_standard_decipher([b"document-id"], params)


def test_standard_security_v1_permissions_can_require_revision_3() -> None:
    # Clear permission bit 9, which ISO 32000-1:2008, Tables 21-22 classify
    # as a revision-3 permission while retaining the V=1 encryption method.
    params: PdfDict = {
        "R": 3,
        "P": -260,
        "O": bytes(32),
        "U": bytes(32),
        "Length": 40,
    }

    config = internal_parse_config([b"document-id"], params, 1, (2, 3))

    assert config.revision == 3


def test_standard_security_v1_revision_2_cannot_clear_revision_3_permissions() -> None:
    params: PdfDict = {
        "R": 2,
        "P": -260,
        "O": bytes(32),
        "U": bytes(32),
        "Length": 40,
    }

    with pytest.raises(ValueError, match="require R=3"):
        internal_parse_config([b"document-id"], params, 1, (2, 3))


def test_standard_security_v1_revision_3_accepts_revision_2_permissions() -> None:
    # Adobe's authorized ISO 32000-1:2008 PDF uses V=1, R=3, and P=-28.
    # Readers need to accept that real-world combination even though no
    # revision-3 permission bit is cleared.
    params: PdfDict = {
        "R": 3,
        "P": -28,
        "O": bytes(32),
        "U": bytes(32),
        "Length": 40,
    }

    config = internal_parse_config([b"document-id"], params, 1, (2, 3))

    assert config.revision == 3


def internal_legacy_security_params(revision: int, permissions: int) -> PdfDict:
    return {
        "R": revision,
        "P": permissions,
        "O": bytes(32),
        "U": bytes(32),
        "Length": 40,
    }


@pytest.mark.parametrize("bit_position", [1, 2])
def test_standard_security_ignores_each_reserved_zero_permission_bit(
    bit_position: int,
) -> None:
    """ISO 32000-1 7.6.3.2: readers ignore every flag outside 3-6 and 9-12.

    Table 22 requires a *writer* to clear bits 1-2, but the same clause binds
    readers the other way, and real producers set them: Distiller writes
    ``/P -9``. Refusing the document denied every page over a bit the reader
    was told to disregard.
    """
    permissions = 0xFFFFFFFC | (1 << (bit_position - 1))

    config = internal_parse_config(
        [b"document-id"],
        internal_legacy_security_params(3, permissions),
        1,
        (2, 3),
    )

    assert config.permissions == permissions


@pytest.mark.parametrize("bit_position", [7, 8, *range(13, 33)])
def test_standard_security_ignores_each_cleared_reserved_one_permission_bit(
    bit_position: int,
) -> None:
    """ISO 32000-1 7.6.3.2: a cleared reserved-one bit is ignored, not fatal."""
    permissions = 0xFFFFFFFC & ~(1 << (bit_position - 1))

    config = internal_parse_config(
        [b"document-id"],
        internal_legacy_security_params(3, permissions),
        1,
        (2, 3),
    )

    assert config.permissions == permissions


@pytest.mark.parametrize("bit_position", [3, 4, 5, 6, 9, 10, 11, 12])
def test_standard_security_accepts_each_clearable_permission_bit(bit_position: int) -> None:
    permissions = 0xFFFFFFFC & ~(1 << (bit_position - 1))
    revision = 3 if bit_position >= 9 else 2

    config = internal_parse_config(
        [b"document-id"],
        internal_legacy_security_params(revision, permissions),
        1,
        (2, 3),
    )

    assert config.permissions == permissions


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("user\u00a0name", "user name"),
        ("I\u00adX", "IX"),
        ("\u00aa", "a"),
        ("\u2168", "IX"),
        ("\u0627\u0628", "\u0627\u0628"),
        ("", ""),
        ("\u00ad", ""),
    ],
)
def test_saslprep_maps_and_normalizes_valid_passwords(value: str, expected: str) -> None:
    assert internal_saslprep(value) == expected


def test_revision_7_uses_revision_6_password_normalization() -> None:
    # ISO/TS 32003:2023, 5.2 applies the ISO 32000-2:2020 revision-6
    # password algorithms to revision 7.
    assert internal_normalize_password("I\u00adX", 7) == b"IX"


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("\u0007", "prohibited character"),
        ("\ue000", "prohibited character"),
        ("\u0627a\u0628", "prohibited character"),
        ("\u0627\u06280", "bidirectional"),
    ],
)
def test_saslprep_rejects_prohibited_and_bidirectionally_invalid_passwords(
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        internal_saslprep(value)


def test_security_handler_uses_explicit_named_crypt_filter() -> None:
    attrs: PdfDict = {
        "Filter": [PdfName.of("Crypt"), PdfName.of("FlateDecode")],
        "DecodeParms": [{"Name": PdfName.of("Special")}, None],
    }

    assert internal_stream_crypt_filter_name(attrs, "Default") == "Special"
    methods: dict[str, internal_CryptMethod] = {"Special": "AESV2"}
    assert internal_resolve_crypt_method("Special", methods) == "AESV2"


def test_security_handler_defaults_explicit_crypt_to_identity() -> None:
    attrs: PdfDict = {"Filter": PdfName.of("Crypt")}

    assert internal_stream_crypt_filter_name(attrs, "Default") == "Identity"


def test_security_handler_rejects_late_crypt_filter() -> None:
    attrs: PdfDict = {"Filter": [PdfName.of("FlateDecode"), PdfName.of("Crypt")]}

    with pytest.raises(PdfParseError, match="first"):
        internal_stream_crypt_filter_name(attrs, "Default")


def test_security_handler_rejects_unknown_named_crypt_filter() -> None:
    with pytest.raises(PdfUnsupportedError, match="Undefined crypt filter"):
        internal_resolve_crypt_method("Unknown", {})


def test_standard_security_parses_embedded_file_filter() -> None:
    params: PdfDict = {
        "CF": {
            "StdCF": {
                "Type": PdfName.of("CryptFilter"),
                "CFM": PdfName.of("AESV2"),
                "AuthEvent": PdfName.of("DocOpen"),
                "Length": 16,
            }
        },
        "StmF": PdfName.of("Identity"),
        "StrF": PdfName.of("Identity"),
        "EFF": PdfName.of("StdCF"),
    }

    _, stream_filter, string_filter, embedded_file_filter, crypt_filters = (
        internal_parse_crypt_filters(params, 4)
    )

    assert stream_filter == "Identity"
    assert string_filter == "Identity"
    assert embedded_file_filter == "StdCF"
    assert crypt_filters == {"StdCF": "AESV2"}


def test_standard_security_defaults_embedded_file_filter_to_stream_filter() -> None:
    params: PdfDict = {
        "CF": {
            "StdCF": {
                "CFM": PdfName.of("AESV3"),
                "AuthEvent": PdfName.of("DocOpen"),
                "Length": 32,
            }
        },
        "StmF": PdfName.of("StdCF"),
    }

    _, stream_filter, _, embedded_file_filter, _ = internal_parse_crypt_filters(params, 5)

    assert stream_filter == "StdCF"
    assert embedded_file_filter == "StdCF"


def test_standard_security_parses_iso_ts_32003_aesv4_configuration() -> None:
    params: PdfDict = {
        "R": 7,
        "P": -4,
        "O": bytes(48),
        "U": bytes(48),
        "OE": bytes(32),
        "UE": bytes(32),
        "Perms": bytes(16),
        "Length": 256,
        "CF": {
            "StdCF": {
                "Type": PdfName.of("CryptFilter"),
                "CFM": PdfName.of("AESV4"),
                "AuthEvent": PdfName.of("DocOpen"),
                "Length": 32,
            }
        },
        "StmF": PdfName.of("StdCF"),
        "StrF": PdfName.of("StdCF"),
    }

    config = internal_parse_config([b"document-id"], params, 6, (7,))

    assert config.version == 6
    assert config.revision == 7
    assert config.length_bits == 256
    assert config.stream_filter == "StdCF"
    assert config.string_filter == "StdCF"
    assert config.crypt_filters == {"StdCF": "AESV4"}


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"CF": {"StdCF": {"CFM": PdfName.of("AESV3"), "Length": 32}}},
        {"CF": {"StdCF": {"CFM": PdfName.of("AESV4"), "Length": 16}}},
    ],
)
def test_standard_security_v6_requires_a_valid_aesv4_filter(params: PdfDict) -> None:
    with pytest.raises(ValueError):
        internal_parse_crypt_filters(params, 6)


def test_standard_security_rejects_undefined_embedded_file_filter() -> None:
    params: PdfDict = {
        "CF": {
            "StdCF": {
                "CFM": PdfName.of("AESV2"),
                "AuthEvent": PdfName.of("DocOpen"),
                "Length": 16,
            }
        },
        "EFF": PdfName.of("Missing"),
    }

    with pytest.raises(ValueError, match="undefined EFF crypt filter"):
        internal_parse_crypt_filters(params, 4)


@pytest.mark.parametrize(
    "filter_config",
    [
        {
            "Type": PdfName.of("WrongType"),
            "CFM": PdfName.of("AESV3"),
            "AuthEvent": PdfName.of("DocOpen"),
            "Length": 32,
        },
        {
            "CFM": PdfName.of("AESV3"),
            "AuthEvent": PdfName.of("EFOpen"),
            "Length": 32,
        },
        {
            "CFM": PdfName.of("AESV3"),
            "AuthEvent": PdfName.of("DocOpen"),
            "Length": 16,
        },
    ],
)
def test_standard_security_rejects_invalid_standard_crypt_filter(
    filter_config: PdfDict,
) -> None:
    params: PdfDict = {
        "CF": {"StdCF": filter_config},
        "StmF": PdfName.of("StdCF"),
    }

    with pytest.raises(ValueError):
        internal_parse_crypt_filters(params, 5)


def test_standard_security_rejects_nonstandard_named_crypt_filter() -> None:
    params: PdfDict = {
        "CF": {
            "Custom": {
                "CFM": PdfName.of("AESV2"),
                "AuthEvent": PdfName.of("DocOpen"),
                "Length": 16,
            }
        }
    }

    with pytest.raises(ValueError, match="unsupported Standard Security crypt filter"):
        internal_parse_crypt_filters(params, 4)


def test_security_handler_uses_eff_for_embedded_file_streams() -> None:
    crypt_filters: dict[str, internal_CryptMethod] = {"StdCF": "AESV2"}
    config = internal_StandardSecurityConfig(
        version=4,
        revision=4,
        permissions=(1 << 32) - 4,
        owner_entry=bytes(32),
        user_entry=bytes(32),
        length_bits=128,
        document_id=b"document-id",
        encrypt_metadata=True,
        stream_filter="Identity",
        string_filter="Identity",
        embedded_file_filter="StdCF",
        crypt_filters=crypt_filters,
        owner_encrypted_key=b"",
        user_encrypted_key=b"",
        encrypted_permissions=b"",
    )
    handler = internal_StandardSecurityHandler(config, bytes(range(16)))
    object_number = 7
    generation_number = 0
    initialization_vector = bytes(range(16))
    plaintext = b"embedded file payload"
    object_key = handler.object_key(object_number, generation_number, b"sAlT")
    ciphertext = initialization_vector + internal_aes_cbc_encrypt(
        object_key,
        initialization_vector,
        plaintext,
        use_padding=True,
    )

    decrypted = handler.decrypt(
        object_number,
        generation_number,
        ciphertext,
        {"Type": PdfName.of("EmbeddedFile")},
    )
    ordinary_stream = handler.decrypt(
        object_number,
        generation_number,
        ciphertext,
        {"Type": PdfName.of("XObject")},
    )

    assert decrypted == plaintext
    assert ordinary_stream == ciphertext


def test_reserved_permission_bits_do_not_block_real_producer_output() -> None:
    """ISO 32000-1 7.6.3.2, end to end on files Distiller actually produced.

    These carry ``/P -9`` (bits 1 and 2 set) with V=2/R=3. Enforcing Table 22's
    writer requirements against them denied every page of a readable document.
    """
    from core_pdf import PdfDocument as PublicDocument

    sample = require_fixture(
        FIXTURES / "llama_index" / "docs" / "examples" / "data" / "10k" / "uber_2021.pdf",
        "llama_index fixtures are not present",
    )

    with PublicDocument(str(sample)) as document:
        assert len(document.pages) > 0
        lines = document.pages[0].get_text_lines()
        assert any("SECURITIES" in line.reconstructed_text().text for line in lines)
