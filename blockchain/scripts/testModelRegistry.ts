import { network } from "hardhat";

async function main() {
  const { ethers } = await network.connect();

const contractAddress =
  "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512";
  const modelRegistry = await ethers.getContractAt(
    "ModelRegistry",
    contractAddress
  );

  const tx = await modelRegistry.submitModelUpdate("TEST_CID_123");

  await tx.wait();

  console.log("Model update submitted!");
  console.log("Transaction hash:", tx.hash);

  const update = await modelRegistry.getModelUpdate(0);

  console.log("CID:", update[0]);
  console.log("Submitted by:", update[1]);
  console.log("Timestamp:", update[2].toString());
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});