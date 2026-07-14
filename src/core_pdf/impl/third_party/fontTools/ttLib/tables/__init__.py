from __future__ import annotations

from types import ModuleType


def get_table_module(py_tag: str) -> ModuleType | None:
    match py_tag:
        case "B_A_S_E_":
            from . import B_A_S_E_

            return B_A_S_E_
        case "C_B_D_T_":
            from . import C_B_D_T_

            return C_B_D_T_
        case "C_B_L_C_":
            from . import C_B_L_C_

            return C_B_L_C_
        case "C_F_F_":
            from . import C_F_F_

            return C_F_F_
        case "C_F_F__2":
            from . import C_F_F__2

            return C_F_F__2
        case "C_O_L_R_":
            from . import C_O_L_R_

            return C_O_L_R_
        case "C_P_A_L_":
            from . import C_P_A_L_

            return C_P_A_L_
        case "D_S_I_G_":
            from . import D_S_I_G_

            return D_S_I_G_
        case "D__e_b_g":
            from . import D__e_b_g

            return D__e_b_g
        case "E_B_D_T_":
            from . import E_B_D_T_

            return E_B_D_T_
        case "E_B_L_C_":
            from . import E_B_L_C_

            return E_B_L_C_
        case "F_F_T_M_":
            from . import F_F_T_M_

            return F_F_T_M_
        case "F__e_a_t":
            from . import F__e_a_t

            return F__e_a_t
        case "G_D_E_F_":
            from . import G_D_E_F_

            return G_D_E_F_
        case "G_P_O_S_":
            from . import G_P_O_S_

            return G_P_O_S_
        case "G_S_U_B_":
            from . import G_S_U_B_

            return G_S_U_B_
        case "G_V_A_R_":
            from . import G_V_A_R_

            return G_V_A_R_
        case "G__l_a_t":
            from . import G__l_a_t

            return G__l_a_t
        case "G__l_o_c":
            from . import G__l_o_c

            return G__l_o_c
        case "H_V_A_R_":
            from . import H_V_A_R_

            return H_V_A_R_
        case "I_F_T_":
            from . import I_F_T_

            return I_F_T_
        case "I_F_T_X_":
            from . import I_F_T_X_

            return I_F_T_X_
        case "J_S_T_F_":
            from . import J_S_T_F_

            return J_S_T_F_
        case "L_T_S_H_":
            from . import L_T_S_H_

            return L_T_S_H_
        case "M_A_T_H_":
            from . import M_A_T_H_

            return M_A_T_H_
        case "M_V_A_R_":
            from . import M_V_A_R_

            return M_V_A_R_
        case "O_S_2f_2":
            from . import O_S_2f_2

            return O_S_2f_2
        case "S_T_A_T_":
            from . import S_T_A_T_

            return S_T_A_T_
        case "S_V_G_":
            from . import S_V_G_

            return S_V_G_
        case "S__i_l_f":
            from . import S__i_l_f

            return S__i_l_f
        case "S__i_l_l":
            from . import S__i_l_l

            return S__i_l_l
        case "T_S_I_B_":
            from . import T_S_I_B_

            return T_S_I_B_
        case "T_S_I_C_":
            from . import T_S_I_C_

            return T_S_I_C_
        case "T_S_I_D_":
            from . import T_S_I_D_

            return T_S_I_D_
        case "T_S_I_J_":
            from . import T_S_I_J_

            return T_S_I_J_
        case "T_S_I_P_":
            from . import T_S_I_P_

            return T_S_I_P_
        case "T_S_I_S_":
            from . import T_S_I_S_

            return T_S_I_S_
        case "T_S_I_V_":
            from . import T_S_I_V_

            return T_S_I_V_
        case "T_S_I__0":
            from . import T_S_I__0

            return T_S_I__0
        case "T_S_I__1":
            from . import T_S_I__1

            return T_S_I__1
        case "T_S_I__2":
            from . import T_S_I__2

            return T_S_I__2
        case "T_S_I__3":
            from . import T_S_I__3

            return T_S_I__3
        case "T_S_I__5":
            from . import T_S_I__5

            return T_S_I__5
        case "T_T_F_A_":
            from . import T_T_F_A_

            return T_T_F_A_
        case "V_A_R_C_":
            from . import V_A_R_C_

            return V_A_R_C_
        case "V_D_M_X_":
            from . import V_D_M_X_

            return V_D_M_X_
        case "V_O_R_G_":
            from . import V_O_R_G_

            return V_O_R_G_
        case "V_V_A_R_":
            from . import V_V_A_R_

            return V_V_A_R_
        case "_a_n_k_r":
            from . import _a_n_k_r

            return _a_n_k_r
        case "_a_v_a_r":
            from . import _a_v_a_r

            return _a_v_a_r
        case "_b_g_c_l":
            from . import _b_g_c_l

            return _b_g_c_l
        case "_b_s_l_n":
            from . import _b_s_l_n

            return _b_s_l_n
        case "_c_i_d_g":
            from . import _c_i_d_g

            return _c_i_d_g
        case "_c_m_a_p":
            from . import _c_m_a_p

            return _c_m_a_p
        case "_c_v_a_r":
            from . import _c_v_a_r

            return _c_v_a_r
        case "_c_v_t":
            from . import _c_v_t

            return _c_v_t
        case "_f_e_a_t":
            from . import _f_e_a_t

            return _f_e_a_t
        case "_f_p_g_m":
            from . import _f_p_g_m

            return _f_p_g_m
        case "_f_v_a_r":
            from . import _f_v_a_r

            return _f_v_a_r
        case "_g_a_s_p":
            from . import _g_a_s_p

            return _g_a_s_p
        case "_g_c_i_d":
            from . import _g_c_i_d

            return _g_c_i_d
        case "_g_l_y_f":
            from . import _g_l_y_f

            return _g_l_y_f
        case "_g_v_a_r":
            from . import _g_v_a_r

            return _g_v_a_r
        case "_h_d_m_x":
            from . import _h_d_m_x

            return _h_d_m_x
        case "_h_e_a_d":
            from . import _h_e_a_d

            return _h_e_a_d
        case "_h_h_e_a":
            from . import _h_h_e_a

            return _h_h_e_a
        case "_h_m_t_x":
            from . import _h_m_t_x

            return _h_m_t_x
        case "_k_e_r_n":
            from . import _k_e_r_n

            return _k_e_r_n
        case "_l_c_a_r":
            from . import _l_c_a_r

            return _l_c_a_r
        case "_l_o_c_a":
            from . import _l_o_c_a

            return _l_o_c_a
        case "_l_t_a_g":
            from . import _l_t_a_g

            return _l_t_a_g
        case "_m_a_x_p":
            from . import _m_a_x_p

            return _m_a_x_p
        case "_m_e_t_a":
            from . import _m_e_t_a

            return _m_e_t_a
        case "_m_o_r_t":
            from . import _m_o_r_t

            return _m_o_r_t
        case "_m_o_r_x":
            from . import _m_o_r_x

            return _m_o_r_x
        case "_n_a_m_e":
            from . import _n_a_m_e

            return _n_a_m_e
        case "_o_p_b_d":
            from . import _o_p_b_d

            return _o_p_b_d
        case "_p_o_s_t":
            from . import _p_o_s_t

            return _p_o_s_t
        case "_p_r_e_p":
            from . import _p_r_e_p

            return _p_r_e_p
        case "_p_r_o_p":
            from . import _p_r_o_p

            return _p_r_o_p
        case "_s_b_i_x":
            from . import _s_b_i_x

            return _s_b_i_x
        case "_t_r_a_k":
            from . import _t_r_a_k

            return _t_r_a_k
        case "_v_h_e_a":
            from . import _v_h_e_a

            return _v_h_e_a
        case "_v_m_t_x":
            from . import _v_m_t_x

            return _v_m_t_x
        case _:
            return None
