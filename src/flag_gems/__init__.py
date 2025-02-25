import torch

from . import testing  # noqa: F401
from . import runtime
from .fused import *  # noqa: F403
from .ops import *  # noqa: F403
from .runtime.commom_utils import Autograd
from .runtime.register import Register

__version__ = "2.2"
device = runtime.device.name
aten_lib = torch.library.Library("aten", "IMPL")
registrar = Register
current_work_registrar = None
runtime.replace_customized_ops(globals())


def enable(lib=aten_lib, unused=None, registrar=registrar):
    global current_work_registrar
    current_work_registrar = registrar(
        (
            ("abs", abs, Autograd.disable),
            ("add.Tensor", add, Autograd.disable),
            ("addmm", addmm, Autograd.disable),
            ("arange.start_step", arange_start, Autograd.disable),
            ("arange.start", arange_start, Autograd.disable),
            ("arange", arange, Autograd.disable),
            ("batch_norm", batch_norm, Autograd.enable),
            ("bitwise_and.Tensor", bitwise_and_tensor, Autograd.disable),
            ("bitwise_and.Scalar", bitwise_and_scalar, Autograd.disable),
            ("bitwise_and.Scalar_Tensor", bitwise_and_scalar_tensor, Autograd.disable),
            ("bitwise_not", bitwise_not, Autograd.disable),
            ("bitwise_or.Tensor", bitwise_or_tensor, Autograd.disable),
            ("bitwise_or.Scalar", bitwise_or_scalar, Autograd.disable),
            ("bitwise_or.Scalar_Tensor", bitwise_or_scalar_tensor, Autograd.disable),
            ("bmm", bmm, Autograd.disable),
            ("clamp", clamp, Autograd.disable),
            ("clamp.Tensor", clamp_tensor, Autograd.disable),
            ("cos", cos, Autograd.disable),
            ("pad", pad, Autograd.disable),
            ("constant_pad_nd", constant_pad_nd, Autograd.disable),
            ("cumsum", cumsum, Autograd.disable),
            ("cummin", cummin, Autograd.disable),
            ("div.Tensor", true_divide, Autograd.disable),
            ("div.Scalar", true_divide, Autograd.disable),
            ("div.Tensor_mode", div_mode, Autograd.disable),
            ("div.Scalar_mode", div_mode, Autograd.disable),
            (
                "divide.Tensor",
                true_divide,
                Autograd.disable,
            ),  # divide, an alias for div
            ("divide.Scalar", true_divide, Autograd.disable),
            ("divide.Tensor_mode", div_mode, Autograd.disable),
            ("divide.Scalar_mode", div_mode, Autograd.disable),
            (
                "true_divide.Tensor",
                true_divide,
                Autograd.disable,
            ),  # true_divide, an alias for div
            ("true_divide.Scalar", true_divide, Autograd.disable),
            ("floor_divide", floor_divide, Autograd.disable),
            ("floor_divide.Scalar", floor_divide, Autograd.disable),
            ("remainder.Tensor", remainder, Autograd.disable),
            ("native_dropout", native_dropout, Autograd.enable),
            ("erf", erf, Autograd.disable),
            ("embedding", embedding, Autograd.enable),
            ("eq.Tensor", eq, Autograd.disable),
            ("eq.Scalar", eq_scalar, Autograd.disable),
            ("exp", exp, Autograd.disable),
            ("exponential_", exponential_, Autograd.disable),
            ("ge.Tensor", ge, Autograd.disable),
            ("ge.Scalar", ge_scalar, Autograd.disable),
            ("gelu", gelu, Autograd.enable),
            ("native_group_norm", group_norm, Autograd.enable),
            ("_weight_norm_interface", weight_norm_interface, Autograd.enable),
            ("_weight_norm", weight_norm, Autograd.enable),
            ("gt.Tensor", gt, Autograd.disable),
            ("gt.Scalar", gt_scalar, Autograd.disable),
            ("instance_norm", instance_norm, Autograd.enable),
            ("isfinite", isfinite, Autograd.disable),
            ("isin.Tensor_Tensor", isin, Autograd.disable),
            ("isin.Scalar_Tensor", isin, Autograd.disable),
            ("isin.Tensor_Scalar", isin, Autograd.disable),
            ("isinf", isinf, Autograd.disable),
            ("isnan", isnan, Autograd.disable),
            ("minimum", minimum, Autograd.disable),
            ("maximum", maximum, Autograd.disable),
            ("native_layer_norm", layer_norm, Autograd.enable),
            ("le.Tensor", le, Autograd.disable),
            ("le.Scalar", le_scalar, Autograd.disable),
            ("lt.Tensor", lt, Autograd.disable),
            ("lt.Scalar", lt_scalar, Autograd.disable),
            ("rms_norm", rms_norm, Autograd.disable),
            ("rand", rand, Autograd.disable),
            ("randn", randn, Autograd.disable),
            ("rand_like", rand_like, Autograd.disable),
            ("randn_like", randn_like, Autograd.disable),
            ("zeros", zeros, Autograd.disable),
            ("ones", ones, Autograd.disable),
            ("full", full, Autograd.disable),
            ("zeros_like", zeros_like, Autograd.disable),
            ("ones_like", ones_like, Autograd.disable),
            ("full_like", full_like, Autograd.disable),
            ("resolve_neg", resolve_neg, Autograd.disable),
            ("resolve_conj", resolve_conj, Autograd.disable),
            ("normal.Tensor_float", normal_tensor_float, Autograd.disable),
            ("normal.float_Tensor", normal_float_tensor, Autograd.disable),
            ("normal.Tensor_Tensor", normal_tensor_tensor, Autograd.disable),
            ("uniform_", uniform_, Autograd.disable),
            ("mean", mean, Autograd.disable),
            ("mean.dim", mean_dim, Autograd.disable),
            ("mm", mm, Autograd.disable),
            ("mul.Tensor", mul, Autograd.disable),
            ("multinomial", multinomial, Autograd.disable),
            ("mv", mv, Autograd.disable),
            ("ne.Tensor", ne, Autograd.disable),
            ("ne.Scalar", ne_scalar, Autograd.disable),
            ("neg", neg, Autograd.disable),
            ("pow.Scalar", pow_scalar, Autograd.disable),
            ("pow.Tensor_Scalar", pow_tensor_scalar, Autograd.disable),
            ("pow.Tensor_Tensor", pow_tensor_tensor, Autograd.disable),
            ("reciprocal", reciprocal, Autograd.disable),
            ("relu", relu, Autograd.enable),
            ("rsqrt", rsqrt, Autograd.disable),
            ("sigmoid", sigmoid, Autograd.enable),
            ("silu", silu, Autograd.enable),
            ("sin", sin, Autograd.disable),
            ("softmax.int", softmax, Autograd.enable),
            ("sort", sort, Autograd.disable),
            ("sub.Tensor", sub, Autograd.disable),
            ("tanh", tanh, Autograd.enable),
            ("triu", triu, Autograd.disable),
            # ("topk", topk, Autograd.disable),
            ("var_mean.correction", var_mean, Autograd.disable),
            ("linalg_vector_norm", vector_norm, Autograd.disable),
            ("where.self_out", where_self_out, Autograd.disable),
            ("where.self", where_self, Autograd.disable),
            ("where.ScalarSelf", where_scalar_self, Autograd.disable),
            ("where.ScalarOther", where_scalar_other, Autograd.disable),
            ("max", max, Autograd.disable),
            ("max.dim", max_dim, Autograd.disable),
            ("min", min, Autograd.disable),
            ("min.dim", min_dim, Autograd.disable),
            ("amax", amax, Autograd.disable),
            ("argmax", argmax, Autograd.disable),
            ("argmin", argmin, Autograd.disable),
            ("prod", prod, Autograd.disable),
            ("prod.dim_int", prod_dim, Autograd.disable),
            ("sum", sum, Autograd.disable),
            ("sum.dim_IntList", sum_dim, Autograd.disable),
            (
                "scaled_dot_product_attention",
                scaled_dot_product_attention,
                Autograd.disable,
            ),
            ("all", all, Autograd.disable),
            ("all.dim", all_dim, Autograd.disable),
            ("all.dims", all_dims, Autograd.disable),
            ("any", any, Autograd.disable),
            ("any.dim", any_dim, Autograd.disable),
            ("any.dims", any_dims, Autograd.disable),
            ("quantile", quantile, Autograd.disable),
            ("log_softmax.int", log_softmax, Autograd.enable),
            ("outer", outer, Autograd.enable),
            ("cross_entropy_loss", cross_entropy_loss, Autograd.enable),
            ("nll_loss_forward", nll_loss_forward, Autograd.disable),
            ("nll_loss_backward", nll_loss_backward, Autograd.disable),
            ("nll_loss2d_forward", nll_loss2d_forward, Autograd.disable),
            ("nll_loss2d_backward", nll_loss2d_backward, Autograd.disable),
            ("scatter.src", scatter, Autograd.disable),
            ("scatter.reduce", scatter, Autograd.disable),
            ("gather", gather, Autograd.disable),
            ("gather_backward", gather_backward, Autograd.disable),
            ("isclose", isclose, Autograd.disable),
            ("allclose", allclose, Autograd.disable),
            ("fill.Scalar", fill_scalar, Autograd.disable),
            ("fill.Tensor", fill_tensor, Autograd.disable),
            ("flip", flip, Autograd.disable),
            ("slice_scatter", slice_scatter, Autograd.disable),
            ("select_scatter", select_scatter, Autograd.disable),
            ("index_select", index_select, Autograd.disable),
            ("tile", tile, Autograd.disable),
            ("masked_fill.Tensor", masked_fill, Autograd.disable),
            ("masked_fill.Scalar", masked_fill, Autograd.disable),
            ("masked_fill_.Tensor", masked_fill_, Autograd.disable),
            ("masked_fill_.Scalar", masked_fill_, Autograd.disable),
            ("_unique2", _unique2, Autograd.disable),
            ("_upsample_bicubic2d_aa", _upsample_bicubic2d_aa, Autograd.disable),
            ("upsample_nearest2d", upsample_nearest2d, Autograd.disable),
            ("nonzero", nonzero, Autograd.disable),
            ("repeat", repeat, Autograd.disable),
            ("masked_select", masked_select, Autograd.disable),
            ("stack", stack, Autograd.disable),
            ("hstack", hstack, Autograd.disable),
            ("cat", cat, Autograd.disable),
            (
                "repeat_interleave.self_int",
                repeat_interleave_self_int,
                Autograd.disable,
            ),
            ("vstack", vstack, Autograd.disable),
            ("repeat_interleave.Tensor", repeat_interleave_tensor, Autograd.disable),
            (
                "repeat_interleave.self_Tensor",
                repeat_interleave_self_tensor,
                Autograd.disable,
            ),
            ("randperm", randperm, Autograd.disable),
            ("diag", diag, Autograd.disable),
            ("diag_embed", diag_embed, Autograd.disable),
            ("diagonal_backward", diagonal_backward, Autograd.disable),
            ("index_add", index_add, Autograd.disable),
            ("count_nonzero", count_nonzero, Autograd.disable),
            ("logical_or", logical_or, Autograd.disable),
            ("logical_and", logical_and, Autograd.disable),
            ("logical_xor", logical_xor, Autograd.disable),
            ("logical_not", logical_not, Autograd.disable),
            ("dot", dot, Autograd.disable),
            ("index_put", index_put, Autograd.disable),
            ("log_sigmoid", log_sigmoid, Autograd.disable),
            ("vdot", vdot, Autograd.disable),
            ("mse_loss", mse_loss, Autograd.disable),
        ),
        user_unused_ops_list=[] if unused is None else unused,
        lib=lib,
    )


class use_gems:
    def __init__(self, unused=None):
        self.lib = torch.library.Library("aten", "IMPL")
        self.unused = [] if unused is None else unused
        self.registrar = Register

    def __enter__(self):
        enable(lib=self.lib, unused=self.unused, registrar=self.registrar)

    def __exit__(self, exc_type, exc_val, exc_tb):
        global current_work_registrar
        del self.lib
        del self.unused
        del self.registrar
        del current_work_registrar


def all_ops():
    return current_work_registrar.get_all_ops()


__all__ = [
    "enable",
    "use_gems",
]
