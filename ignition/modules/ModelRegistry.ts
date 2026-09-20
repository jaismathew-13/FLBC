import { buildModule } from "@nomicfoundation/hardhat-ignition/modules";

const ModelRegistryModule = buildModule("ModelRegistryModule", (m) => {
  const modelRegistry = m.contract("ModelRegistry");

  return { modelRegistry };
});

export default ModelRegistryModule;