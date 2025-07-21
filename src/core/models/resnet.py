from ..batchnorm import BatchNorm2d
from ..modules import Conv2D, Linear, Module, ReLU, Sequential
from ..ops import add
from ..pool import AdaptiveAvgPool2d
from ..tensor import Tensor


class BasicBlock(Module):
  expansion = 1

  def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
    super().__init__()
    self.conv1 = Conv2D(
      in_channels, out_channels, 3, stride=stride, padding=1, bias=False
    )
    self.bn1 = BatchNorm2d(out_channels)
    self.relu = ReLU()
    self.conv2 = Conv2D(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
    self.bn2 = BatchNorm2d(out_channels)

    self.shortcut = Sequential()
    if stride != 1 or in_channels != out_channels:
      self.shortcut = Sequential(
        Conv2D(in_channels, out_channels, 1, stride=stride, bias=False),
        BatchNorm2d(out_channels),
      )

  def __call__(self, x: Tensor) -> Tensor:
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out = add(out, self.shortcut(x))
    return self.relu(out)


class Bottleneck(Module):
  expansion = 4

  def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
    super().__init__()
    width = out_channels
    self.conv1 = Conv2D(in_channels, width, 1, bias=False)
    self.bn1 = BatchNorm2d(width)
    self.conv2 = Conv2D(width, width, 3, stride=stride, padding=1, bias=False)
    self.bn2 = BatchNorm2d(width)
    self.conv3 = Conv2D(width, out_channels * self.expansion, 1, bias=False)
    self.bn3 = BatchNorm2d(out_channels * self.expansion)
    self.relu = ReLU()

    self.shortcut = Sequential()
    if stride != 1 or in_channels != out_channels * self.expansion:
      self.shortcut = Sequential(
        Conv2D(
          in_channels, out_channels * self.expansion, 1, stride=stride, bias=False
        ),
        BatchNorm2d(out_channels * self.expansion),
      )

  def __call__(self, x: Tensor) -> Tensor:
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.relu(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    out = add(out, self.shortcut(x))
    return self.relu(out)


class ResNet(Module):
  def __init__(
    self,
    block: type[Module],
    layers: list[int],
    num_classes: int = 1000,
    in_channels: int = 1,
  ):
    super().__init__()

    self.input_channels = 64
    self.conv1 = Conv2D(in_channels, 64, 7, stride=2, padding=3, bias=False)
    self.bn1 = BatchNorm2d(64)
    self.relu = ReLU()

    self.layer1 = self._make_layer(block, 64, layers[0], 1)
    self.layer2 = self._make_layer(block, 128, layers[1], 2)
    self.layer3 = self._make_layer(block, 256, layers[2], 2)
    self.layer4 = self._make_layer(block, 512, layers[3], 2)

    self.avgpool = AdaptiveAvgPool2d((1, 1))
    self.fc = Linear(512 * block.expansion, num_classes)

    for p in self.parameters():
      p.data = p.data.astype("float32")

  def _make_layer(
    self, block: type[Module], out_c: int, blocks: int, stride: int
  ) -> Sequential:
    strides = [stride] + [1] * (blocks - 1)
    layers: list[Module] = []
    for s in strides:
      layers.append(block(self.input_channels, out_c, s))
      self.input_channels = out_c * block.expansion
    return Sequential(*layers)

  def __call__(self, x: Tensor) -> Tensor:
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.avgpool(out)
    out = out.reshape(out.shape[0], -1)
    return self.fc(out)


_SPECS: dict[int, tuple[type[Module], list[int]]] = {
  18: (BasicBlock, [2, 2, 2, 2]),
  34: (BasicBlock, [3, 4, 6, 3]),
  50: (Bottleneck, [3, 4, 6, 3]),
  101: (Bottleneck, [3, 4, 23, 3]),
  152: (Bottleneck, [3, 8, 36, 3]),
}


def resnet(depth: int, **kw) -> ResNet:
  if depth not in _SPECS:
    raise ValueError(
      f"Unsupported ResNet depth. The supported depths are: {list(_SPECS.keys())}."
    )
  block, cfg = _SPECS[depth]
  return ResNet(block, cfg, **kw)


def resnet18(**kw) -> ResNet:
  return resnet(18, **kw)


def resnet34(**kw) -> ResNet:
  return resnet(34, **kw)


def resnet50(**kw) -> ResNet:
  return resnet(50, **kw)


def resnet101(**kw) -> ResNet:
  return resnet(101, **kw)


def resnet152(**kw) -> ResNet:
  return resnet(152, **kw)


__all__ = [
  "BasicBlock",
  "Bottleneck",
  "ResNet",
  "resnet",
  "resnet18",
  "resnet34",
  "resnet50",
  "resnet101",
  "resnet152",
]
