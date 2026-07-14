# SPDX-License-Identifier: AGPL-3.0-only
import struct
from collections.abc import Callable, Sequence
from hashlib import md5, sha256, sha384, sha512

from core_pdf.impl.engine.spec.s_07_security.aes import AES
from core_pdf.impl.engine.spec.s_07_security.crypto_constants import PDF_PADDING
from core_pdf.impl.engine.spec.s_07_security.rc4 import CryptRC4
from core_pdf.impl.engine.spec.s_07_security.saslprep import saslprep
from core_pdf.impl.engine.spec.s_07_syntax.errors import PdfUnsupportedError
from core_pdf.impl.engine.spec.s_07_syntax.primitives import (
    PdfDictLike,
    PdfObject,
    coerce_to_bytes,
    parse_int,
    parse_name,
)

DecryptFn = Callable[[int, int, bytes], bytes]


class PDFEncryptionError(Exception):
    """Exception raised for PDF encryption errors."""

    pass


class PDFPasswordIncorrect(PDFEncryptionError):
    """Exception raised when an incorrect password is provided."""

    pass


def get_int(val: PdfObject, default: int = 0) -> int:
    parsed = parse_int(val, default if val is None else None)
    if parsed is None:
        raise PDFEncryptionError(f"invalid integer value: {val!r}")
    return parsed


def get_uint(val: PdfObject, n_bits: int = 32) -> int:
    v = get_int(val, 0)
    if v >= 0:
        return v
    return v + (1 << n_bits)


def get_name(val: PdfObject) -> str:
    return parse_name(val, "") or ""


class PdfStandardSecurityHandler:
    supported_revisions: tuple[int, ...] = (2, 3)

    def __init__(
        self,
        docid: Sequence[PdfObject],
        param: PdfDictLike,
        password: str = "",
    ) -> None:
        self.docid = docid
        self.param = param
        self.password = password
        self.encrypt_metadata = True
        self.init()

    def init(self) -> None:
        self.init_params()
        if self.r not in self.supported_revisions:
            raise PDFEncryptionError(f"Unsupported revision: param={self.param!r}")
        self.init_key()

    def init_params(self) -> None:
        self.v = get_int(self.param.get("V", 0))
        self.r = get_int(self.param.get("R"))
        self.p = get_uint(self.param.get("P"), 32)
        self.o = coerce_to_bytes(self.param.get("O"))
        self.u = coerce_to_bytes(self.param.get("U"))
        self.length = get_int(self.param.get("Length", 40))

    def init_key(self) -> None:
        self.key = self.authenticate(self.password)
        if self.key is None:
            raise PDFPasswordIncorrect("Incorrect password")

    def is_printable(self) -> bool:
        return bool(self.p & 4)

    def is_modifiable(self) -> bool:
        return bool(self.p & 8)

    def is_extractable(self) -> bool:
        return bool(self.p & 16)

    def compute_u(self, key: bytes) -> bytes:
        if self.r == 2:
            return CryptRC4(key).encrypt(PDF_PADDING)
        else:
            h = md5(PDF_PADDING)
            docid_list = self.docid
            first_id = coerce_to_bytes(docid_list[0]) if docid_list and len(docid_list) > 0 else b""
            h.update(first_id)
            result = CryptRC4(key).encrypt(h.digest())
            for i in range(1, 20):
                k = bytes(c ^ i for c in key)
                result = CryptRC4(k).encrypt(result)
            result += result
            return result

    def compute_encryption_key(self, password: bytes) -> bytes:
        password = (password + PDF_PADDING)[:32]
        h = md5(password)
        h.update(self.o)
        h.update(struct.pack("<L", self.p))
        docid_list = self.docid
        first_id = coerce_to_bytes(docid_list[0]) if docid_list and len(docid_list) > 0 else b""
        h.update(first_id)
        if self.r >= 4 and not self.encrypt_metadata:
            h.update(b"\xff\xff\xff\xff")
        result = h.digest()
        n = 5
        if self.r >= 3:
            n = self.length // 8
            for _ in range(50):
                result = md5(result[:n]).digest()
        return result[:n]

    def authenticate(self, password: str) -> bytes | None:
        password_bytes = password.encode("latin1")
        key = self.authenticate_user_password(password_bytes)
        if key is None:
            key = self.authenticate_owner_password(password_bytes)
        return key

    def authenticate_user_password(self, password: bytes) -> bytes | None:
        key = self.compute_encryption_key(password)
        if self.verify_encryption_key(key):
            return key
        return None

    def verify_encryption_key(self, key: bytes) -> bool:
        u = self.compute_u(key)
        if self.r == 2:
            return u == self.u
        return u[:16] == self.u[:16]

    def authenticate_owner_password(self, password: bytes) -> bytes | None:
        password = (password + PDF_PADDING)[:32]
        h = md5(password)
        if self.r >= 3:
            for _ in range(50):
                h = md5(h.digest())
        n = 5
        if self.r >= 3:
            n = self.length // 8
        key = h.digest()[:n]
        if self.r == 2:
            user_password = CryptRC4(key).decrypt(self.o)
        else:
            user_password = self.o
            for i in range(19, -1, -1):
                k = bytes(c ^ i for c in key)
                user_password = CryptRC4(k).decrypt(user_password)
        return self.authenticate_user_password(user_password)

    def decrypt(
        self,
        objid: int,
        genno: int,
        data: bytes,
        attrs: PdfDictLike | None = None,
    ) -> bytes:
        return self.decrypt_rc4(objid, genno, data)

    def decrypt_rc4(self, objid: int, genno: int, data: bytes) -> bytes:
        assert self.key is not None
        key = self.key + struct.pack("<L", objid)[:3] + struct.pack("<L", genno)[:2]
        h = md5(key)
        key = h.digest()[: min(len(key), 16)]
        return CryptRC4(key).decrypt(data)


class PdfStandardSecurityHandlerV4(PdfStandardSecurityHandler):
    supported_revisions: tuple[int, ...] = (4,)

    def init_params(self) -> None:
        super().init_params()
        self.length = 128
        cf = self.param.get("CF")
        if cf is None:
            self.cf = {}
        elif not isinstance(cf, dict):
            raise PDFEncryptionError("Invalid crypt filter dictionary: CF")
        else:
            self.cf = cf
        self.stmf = get_name(self.param.get("StmF", "Identity")) or "Identity"
        self.strf = get_name(self.param.get("StrF", "Identity")) or "Identity"
        self.encrypt_metadata = bool(self.param.get("EncryptMetadata", True))
        self.cfm = {}
        for k, v in self.cf.items():
            if not isinstance(v, dict):
                raise PDFEncryptionError(f"Invalid crypt filter dictionary: {k!r}")
            f = self.get_cfm(get_name(v.get("CFM", "")))
            if f is None:
                raise PDFEncryptionError(f"Unknown crypt filter method CFM: {v.get('CFM')!r}")
            self.cfm[k] = f
        if self.strf != "Identity" and self.strf not in self.cfm:
            raise PDFEncryptionError(f"Undefined crypt filter: {self.strf}")

    def get_cfm(self, name: str) -> DecryptFn | None:
        if name == "V2":
            return self.decrypt_rc4
        elif name == "AESV2":
            return self.decrypt_aes128
        return None

    def decrypt(
        self,
        objid: int,
        genno: int,
        data: bytes,
        attrs: PdfDictLike | None = None,
        name: str | None = None,
    ) -> bytes:
        if not self.encrypt_metadata and attrs is not None:
            t = attrs.get("Type")
            if t is not None and get_name(t) == "Metadata":
                return data
        if name is None:
            name = self.stmf if attrs is not None else self.strf
        if name == "Identity":
            return data
        fn = self.cfm.get(name)
        if fn is None:
            raise PdfUnsupportedError(f"Undefined crypt filter: {name}")
        return fn(objid, genno, data)

    def decrypt_aes128(self, objid: int, genno: int, data: bytes) -> bytes:
        assert self.key is not None
        key = self.key + struct.pack("<L", objid)[:3] + struct.pack("<L", genno)[:2] + b"sAlT"
        h = md5(key)
        key = h.digest()[: min(len(key), 16)]
        initialization_vector = data[:16]
        ciphertext = data[16:]
        cipher = AES(key)
        return cipher.decrypt_cbc(initialization_vector, ciphertext, padding=True)


class PdfStandardSecurityHandlerV5(PdfStandardSecurityHandlerV4):
    supported_revisions = (5, 6)

    def init_params(self) -> None:
        super().init_params()
        self.length = 256
        self.oe = coerce_to_bytes(self.param.get("OE"))
        self.ue = coerce_to_bytes(self.param.get("UE"))
        self.o_hash = self.o[:32]
        self.o_validation_salt = self.o[32:40]
        self.o_key_salt = self.o[40:]
        self.u_hash = self.u[:32]
        self.u_validation_salt = self.u[32:40]
        self.u_key_salt = self.u[40:]

    def get_cfm(self, name: str) -> DecryptFn | None:
        if name == "AESV3":
            return self.decrypt_aes256
        return None

    def authenticate(self, password: str) -> bytes | None:
        password_b = self.normalize_password(password)
        # Owner password validation
        hash_val = self.password_hash(password_b, self.o_validation_salt, self.u)
        if hash_val == self.o_hash:
            hash_val = self.password_hash(password_b, self.o_key_salt, self.u)
            cipher = AES(hash_val)
            return cipher.decrypt_cbc(b"\0" * 16, self.oe, padding=False)
        # User password validation
        hash_val = self.password_hash(password_b, self.u_validation_salt)
        if hash_val == self.u_hash:
            hash_val = self.password_hash(password_b, self.u_key_salt)
            cipher = AES(hash_val)
            return cipher.decrypt_cbc(b"\0" * 16, self.ue, padding=False)
        return None

    def normalize_password(self, password: str) -> bytes:
        if self.r == 6:
            if not password:
                return b""
            password = saslprep(password)
        return password.encode("utf-8")[:127]

    def password_hash(
        self,
        password: bytes,
        salt: bytes,
        vector: bytes | None = None,
    ) -> bytes:
        if self.r == 5:
            return self.r5_password(password, salt, vector)
        return self.r6_password(password, salt[0:8], vector)

    def r5_password(
        self,
        password: bytes,
        salt: bytes,
        vector: bytes | None = None,
    ) -> bytes:
        h = sha256(password)
        h.update(salt)
        if vector is not None:
            h.update(vector)
        return h.digest()

    def r6_password(
        self,
        password: bytes,
        salt: bytes,
        vector: bytes | None = None,
    ) -> bytes:
        initial_hash = sha256(password)
        initial_hash.update(salt)
        if vector is not None:
            initial_hash.update(vector)
        k = initial_hash.digest()
        hashes = (sha256, sha384, sha512)
        round_no = last_byte_val = 0
        while round_no < 64 or last_byte_val > round_no - 32:
            k1 = (password + k + (vector or b"")) * 64
            cipher = AES(k[:16])
            e = cipher.encrypt_cbc(k[16:32], k1, padding=False)
            next_hash = hashes[self.bytes_mod_3(e[:16])]
            k = next_hash(e).digest()
            last_byte_val = e[len(e) - 1]
            round_no += 1
        return k[:32]

    @staticmethod
    def bytes_mod_3(input_bytes: bytes) -> int:
        return sum(b % 3 for b in input_bytes) % 3

    def decrypt_aes256(self, objid: int, genno: int, data: bytes) -> bytes:
        initialization_vector = data[:16]
        ciphertext = data[16:]
        assert self.key is not None
        cipher = AES(self.key)
        return cipher.decrypt_cbc(initialization_vector, ciphertext, padding=True)


SECURITY_HANDLER_REGISTRY = {
    1: PdfStandardSecurityHandler,
    2: PdfStandardSecurityHandler,
    3: PdfStandardSecurityHandler,
    4: PdfStandardSecurityHandlerV4,
    5: PdfStandardSecurityHandlerV5,
    6: PdfStandardSecurityHandlerV5,
}
