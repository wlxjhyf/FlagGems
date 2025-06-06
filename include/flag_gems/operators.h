#pragma once
#include <optional>
#include "torch/torch.h"

namespace flag_gems {
at::Tensor add_tensor(const at::Tensor &a_, const at::Tensor &b_);
at::Tensor sum_dim(const at::Tensor &self,
                   at::OptionalIntArrayRef dim,
                   bool keepdim = false,
                   ::std::optional<at::ScalarType> dtype = ::std::nullopt);
}  // namespace flaggems
