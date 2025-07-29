from math import ceil

from ..batchnorm import BatchNorm2d
from ..modules import Conv2D, Linear, Module, Sequential
from ..ops import add, hard_sigmoid, hard_swish
from ..pool import AdaptiveAvgPool2d
from ..tensor import Tensor

__all__ = [
  "EfficientNet",
  "efficientnet_b0",
  "efficientnet_b1",
  "efficientnet_b2",
  "efficientnet_b3",
  "efficientnet_b4",
  "efficientnet_b5",
  "efficientnet_b6",
  "efficientnet_b7",
]

_BASE_MODEL = [
  [1, 16, 1, 1, 3],
  [6, 24, 2, 2, 3],
  [6, 40, 2, 2, 5],
  [6, 80, 3, 2, 3],
  [6, 112, 3, 1, 5],
  [6, 192, 4, 2, 5],
  [6, 320, 1, 1, 3],
]

_META_PARAMETERS = {
  "B0": (0, 224, 0.2),
  "B1": (0.5, 240, 0.2),
  "B2": (1, 260, 0.3),
  "B3": (2, 300, 0.3),
  "B4": (3, 380, 0.4),
  "B5": (4, 456, 0.4),
  "B6": (5, 528, 0.5),
  "B7": (6, 600, 0.5),
}


def _make_divisible(value: int, divisor: int = 8, minimum: int | None = None) -> int:
  minimum = divisor if minimum is None else minimum
  new_value = max(minimum, int(value + divisor / 2) // divisor * divisor)
  if new_value < 0.9 * value:
    new_value += divisor
  return new_value


class ConvolutionBatchNormalizationActivation(Module):
  def __init__(
    self,
    input_channels: int,
    output_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    groups: int = 1,
  ):
    self.convolution = Conv2D(
      input_channels,
      output_channels,
      kernel_size,
      stride=stride,
      padding=padding,
      bias=False,
      groups=groups,
    )
    self.batch_normalization = BatchNorm2d(output_channels)
    self.activation = hard_swish

  def __call__(self, input_tensor: Tensor) -> Tensor:
    output_tensor = self.convolution(input_tensor)
    output_tensor = self.batch_normalization(output_tensor)
    return self.activation(output_tensor)


class SqueezeExcite(Module):
  def __init__(self, channels: int, reduction_ratio: int):
    reduced_channels = channels // reduction_ratio
    self.pooling = AdaptiveAvgPool2d((1, 1))
    self.reduction_convolution = Conv2D(channels, reduced_channels, 1, bias=False)
    self.reduction_activation = hard_swish
    self.expansion_convolution = Conv2D(reduced_channels, channels, 1, bias=False)
    self.expansion_activation = hard_sigmoid

  def __call__(self, input_tensor: Tensor) -> Tensor:
    scale_tensor = self.pooling(input_tensor)
    scale_tensor = self.reduction_convolution(scale_tensor)
    scale_tensor = self.reduction_activation(scale_tensor)
    scale_tensor = self.expansion_convolution(scale_tensor)
    scale_tensor = self.expansion_activation(scale_tensor)
    return input_tensor * scale_tensor


class InvertedResidualBlock(Module):
  def __init__(
    self,
    input_channels: int,
    output_channels: int,
    kernel_size: int,
    stride: int,
    expansion_ratio: int,
    reduction_ratio: int,
    use_skip: bool,
  ):
    expanded_channels = _make_divisible(input_channels * expansion_ratio)
    self.use_skip = use_skip and stride == 1 and input_channels == output_channels
    self.expand = None
    if expanded_channels != input_channels:
      self.expand = ConvolutionBatchNormalizationActivation(
        input_channels, expanded_channels, 1, 1, 0
      )
    self.depthwise = ConvolutionBatchNormalizationActivation(
      expanded_channels,
      expanded_channels,
      kernel_size,
      stride,
      kernel_size // 2,
      groups=expanded_channels,
    )
    self.squeeze_excite = SqueezeExcite(expanded_channels, reduction_ratio)
    self.project_convolution = Conv2D(expanded_channels, output_channels, 1, bias=False)
    self.project_batch_normalization = BatchNorm2d(output_channels)

  def __call__(self, input_tensor: Tensor) -> Tensor:
    output_tensor = input_tensor
    if self.expand is not None:
      output_tensor = self.expand(output_tensor)
    output_tensor = self.depthwise(output_tensor)
    output_tensor = self.squeeze_excite(output_tensor)
    output_tensor = self.project_convolution(output_tensor)
    output_tensor = self.project_batch_normalization(output_tensor)
    if self.use_skip:
      output_tensor = add(output_tensor, input_tensor)
    return output_tensor


class EfficientNet(Module):
  def __init__(
    self, variant: str, number_of_classes: int = 1000, input_channels: int = 3
  ):
    width_multiplier, depth_multiplier, dropout_probability = self._scaling_factors(
      variant
    )
    last_channels = _make_divisible(1280 * width_multiplier)
    initial_channels = _make_divisible(32 * width_multiplier)
    self.initial_block = ConvolutionBatchNormalizationActivation(
      input_channels, initial_channels, 3, 2, 1
    )
    blocks: list[Module] = []
    current_channels = initial_channels
    for expansion_ratio, base_channels, repeats, stride, kernel_size in _BASE_MODEL:
      output_channels = _make_divisible(base_channels * width_multiplier)
      repeat_count = ceil(repeats * depth_multiplier)
      for repeat_index in range(repeat_count):
        stride_value = stride if repeat_index == 0 else 1
        blocks.append(
          InvertedResidualBlock(
            current_channels,
            output_channels,
            kernel_size,
            stride_value,
            expansion_ratio,
            reduction_ratio=4,
            use_skip=True,
          )
        )
        current_channels = output_channels
    self.blocks = Sequential(*blocks)
    self.final_convolution_block = ConvolutionBatchNormalizationActivation(
      current_channels, last_channels, 1, 1, 0
    )
    self.global_pool = AdaptiveAvgPool2d((1, 1))
    self.classifier_dropout_probability = dropout_probability
    self.classifier = Linear(last_channels, number_of_classes)

  def _scaling_factors(self, variant: str, alpha: float = 1.2, beta: float = 1.1):
    phi, _, drop_rate = _META_PARAMETERS[variant]
    depth_multiplier = alpha**phi
    width_multiplier = beta**phi
    return width_multiplier, depth_multiplier, drop_rate

  def __call__(self, input_tensor: Tensor) -> Tensor:
    output_tensor = self.initial_block(input_tensor)
    output_tensor = self.blocks(output_tensor)
    output_tensor = self.final_convolution_block(output_tensor)
    output_tensor = self.global_pool(output_tensor)
    output_tensor = output_tensor.reshape(output_tensor.shape[0], -1)
    return self.classifier(output_tensor)


def efficientnet_b0(**kwargs) -> EfficientNet:
  return EfficientNet("B0", **kwargs)


def efficientnet_b1(**kwargs) -> EfficientNet:
  return EfficientNet("B1", **kwargs)


def efficientnet_b2(**kwargs) -> EfficientNet:
  return EfficientNet("B2", **kwargs)


def efficientnet_b3(**kwargs) -> EfficientNet:
  return EfficientNet("B3", **kwargs)


def efficientnet_b4(**kwargs) -> EfficientNet:
  return EfficientNet("B4", **kwargs)


def efficientnet_b5(**kwargs) -> EfficientNet:
  return EfficientNet("B5", **kwargs)


def efficientnet_b6(**kwargs) -> EfficientNet:
  return EfficientNet("B6", **kwargs)


def efficientnet_b7(**kwargs) -> EfficientNet:
  return EfficientNet("B7", **kwargs)
