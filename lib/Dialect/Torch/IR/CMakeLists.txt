add_mlir_library(TorchMLIRTorchDialect
  GeneratedTorchOps.cpp
  TorchDialect.cpp
  TorchOps.cpp
  TorchTypes.cpp
  UtilsForGeneratedTorchOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/torch-mlir/Dialect/Torch

  DEPENDS
  MLIRTorchOpsIncGen
  MLIRTorchTypesIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)

torch_mlir_target_includes(TorchMLIRTorchDialect)
