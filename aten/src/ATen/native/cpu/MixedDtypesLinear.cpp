#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <iostream>
#include "c10/util/logging_is_google_glog.h"



namespace at {
namespace native {

template <typename scalar_t, typename scalar_a_t, typename scalar_b_t>
void mixed_gemm_notrans_(
    int64_t m,
    int64_t n,
    int64_t k,
    const scalar_a_t* a,
    int64_t lda,
    const scalar_b_t* b,
    int64_t ldb,
    scalar_t* c,
    int64_t ldc,
    scalar_t* scale) {
  for (const auto l : c10::irange(k)) {
    for (const auto j : c10::irange(n)) {
      scalar_t val = (b[l + j * ldb] - 128) * scale[j];
      for (const auto i : c10::irange(m)) {
        c[i * ldc + j] += a[l + i * lda] * val;
      }
   }
  }
}

template<typename ElementOutput, typename ElementInputA=ElementOutput, typename ElementInputB>  // , typename EpilogueTag>
Tensor
_internal_mixed_dtypes_linear_cpu(
    const Tensor& input, const Tensor& weight, const Tensor& scale,
    const Tensor& bias) {
  const int length_m = input.size(0);
  // auto input_k = input.size(1);
  const int length_k = weight.size(0);
  // auto weight_n = weight.size(1);
  const int length_n = scale.size(0);


  TORCH_CHECK(input.size(1) == length_k, "input size 1");
  TORCH_CHECK(weight.size(1) == length_n, "weight size 1");

  Tensor output = at::zeros({length_m, length_n}, input.options());

  std::cout << "input.sizes() = " << input.sizes() << std::endl;
  std::cout << "weight.sizes() = " << weight.sizes() << std::endl;
  std::cout << "scale.sizes() = " << scale.sizes() << std::endl;

  mixed_gemm_notrans_<ElementOutput, ElementInputA, ElementInputB> (
    length_m,
    length_n,
    length_k,
    input.data_ptr<ElementInputA>(),
    length_m,
    weight.data_ptr<ElementInputB>(),
    length_n,
    output.data_ptr<ElementOutput>(),
    length_n,
    scale.data_ptr<ElementOutput>()
  );

  // ignore bias and activation for now

  return output;
}

template<typename ElementInputA, typename ElementInputB>
Tensor
mixed_dtypes_linear_dispatch_bias_activation_cpu(
    const Tensor& input, const Tensor& weight, const Tensor& scale,
    const Tensor& bias, const c10::string_view& activation) {
    if (bias.numel() == 0) {
      if (activation == "none") {
        return _internal_mixed_dtypes_linear_cpu<
          ElementInputA,
          ElementInputA,
          ElementInputB>(input.contiguous(), weight.contiguous(), scale, bias);
      }
      AT_ERROR("*1 mixed_dtypes_linear_dispatch_bias_activation: Activation \"",
               activation, "\" is not supported");
      return Tensor{};
    }
    else {
      AT_ERROR("*2 mixed_dtypes_linear_dispatch_bias_activation: Activation \"",
               activation, "\" is not supported");
      return Tensor{};
    }
}

Tensor
_mixed_dtypes_linear_cpu(const Tensor& input, const Tensor& weight,
                     const Tensor& scale,
                     const c10::optional<Tensor>& bias_opt,
                     const c10::optional<c10::string_view> activation_opt) {

  const auto bias = bias_opt.has_value() ? *bias_opt : Tensor{};
  const auto activation = activation_opt.has_value() ? *activation_opt : "none";

  // Validate datatypes of input tensors.
  TORCH_CHECK(input.dtype() == at::kHalf ||
              input.dtype() == at::kBFloat16 ||
              input.dtype() == at::kFloat,
              "_mixed_dtypes_linear_cpu: The input datatype ", input.dtype(),
              " is not supported");
  TORCH_CHECK(weight.dtype() == at::kByte,
              "_mixed_dtypes_linear_cpu: The weight datatype ", weight.dtype(),
              " is not supported");
  TORCH_CHECK(scale.dtype() == input.dtype(),
              "_mixed_dtypes_linear_cpu: Expected scale datatype ", input.dtype(),
              " but got", scale.dtype());
  if (bias.numel() != 0) {
    TORCH_CHECK(bias.dtype() == input.dtype(),
                "_mixed_dtypes_linear_cpu: Expected bias datatype ", input.dtype(),
                " but got", bias.dtype());
  }

  // Squash the batch dimensions of the input tensor with its
  // next-to-last dimensions.
  const auto input_sizes = input.sizes().vec();
  const auto input_2d = input.reshape({-1, input_sizes.back()});

  // Validate layouts of input tensors.
  TORCH_CHECK(input_2d.layout() == Layout::Strided,
              "_mixed_dtypes_linear_cpu: Expected input argument to be strided, "
              "but got layout ", input_2d.layout());
  TORCH_CHECK(input_2d.dim() == 2,
              "_mixed_dtypes_linear_cpu: Expected input argument to be 2D tensor, "
              "got ", input_2d.dim(), " dims");
  const auto strides_input = input_2d.strides();
  TORCH_CHECK(strides_input[0] > 1 && strides_input[1] == 1,
              "_mixed_dtypes_linear_cpu: Invalid strides for input argument: row "
              "stride = ", strides_input[0], ", column stride = ",
              strides_input[1]);
  TORCH_CHECK(weight.layout() == Layout::Strided,
              "_mixed_dtypes_linear_cpu: Expected input argument to be strided, "
              "but got layout ", weight.layout());
  TORCH_CHECK(weight.dim() == 2,
              "_mixed_dtypes_linear_cpu: Expected weight argument to be 2D tensor, "
              "got ", weight.dim(), " dims");
  const auto strides_weight = weight.strides();
  TORCH_CHECK(strides_weight[0] > 1 && strides_weight[1] == 1,
              "_mixed_dtypes_linear_cpu: Invalid strides for weight argument: row "
              "stride = ", strides_weight[0], ", column stride = ",
              strides_weight[1]);
  TORCH_CHECK(scale.dim() == 1,
              "_mixed_dtypes_linear_cpu: Expected scale argument to be 1D tensor, "
              "got ", scale.dim(), " dims");
  if (bias.numel() != 0) {
    TORCH_CHECK(bias.dim() == 1,
                "_mixed_dtypes_linear_cpu: Expected bias argument to be 1D ",
                "tensor, got ", bias.dim(), " dims");
  }

  // Validate sizes of input tensors.
  TORCH_CHECK(input_2d.size(1) == weight.size(0),
              "_mixed_dtypes_linear_cpu: Expected input argument to have ",
              weight.size(0), " columns, but got ", input_2d.size(1));
  TORCH_CHECK(weight.size(1) == scale.size(0)  ||
              2 * weight.size(1) == scale.size(0),
              "_mixed_dtypes_linear_cpu: Expected weight argument to have either ",
              scale.size(0), " or ", scale.size(0) / 2.f, " columns, but got ",
              weight.size(1));
  if (bias.numel() != 0) {
      TORCH_CHECK(bias.size(0) == scale.size(0),
                  "_mixed_dtypes_linear_cpu: Expected bias argument to have ",
                  scale.size(0), " elements, but got ", bias.size(0));
  }

  Tensor output;
  auto scalar_type_quant = weight.scalar_type();
  // if (weight.size(1) != scale.size(0)) {
  //   scalar_type_quant = at::ScalarType::QUInt4x2;
  // }
    AT_DISPATCH_SWITCH(
      input.scalar_type(),
      "_mixed_dtypes_linear",
      AT_DISPATCH_CASE(
          at::ScalarType::Half,
          [&]() {
            AT_DISPATCH_SWITCH(
                scalar_type_quant,
                "_mixed_dtypes_linear",
                AT_DISPATCH_CASE(
                    at::ScalarType::Byte,
                    [&]() {
                      output =
                          mixed_dtypes_linear_dispatch_bias_activation_cpu<
                              at::Half,
                              uint8_t>(input_2d, weight, scale, bias,
                                       activation);
                      return;
                    }));
          })
      AT_DISPATCH_CASE(
          at::ScalarType::BFloat16,
          [&]() {
            AT_DISPATCH_SWITCH(
                scalar_type_quant,
                "_mixed_dtypes_linear",
                AT_DISPATCH_CASE(
                    at::ScalarType::Byte,
                    [&]() {
                      output =
                          mixed_dtypes_linear_dispatch_bias_activation_cpu<
                              at::BFloat16,
                              uint8_t>(input_2d, weight, scale, bias,
                                       activation);
                      return;
                    }));
          }));



  auto output_sizes = input_sizes;
  output_sizes.back() = scale.size(0);
  return output.reshape(output_sizes);
}

}  // namespace native
}  // namespace at
