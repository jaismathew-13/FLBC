// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract ModelRegistry {

    struct ModelUpdate {
        string cid;
        address submittedBy;
        uint256 timestamp;
    }

    ModelUpdate[] public modelUpdates;

    function submitModelUpdate(string memory _cid) public {
        modelUpdates.push(
            ModelUpdate(
                _cid,
                msg.sender,
                block.timestamp
            )
        );
    }

    function getModelUpdate(uint256 index)
        public
        view
        returns (
            string memory,
            address,
            uint256
        )
    {
        ModelUpdate memory update = modelUpdates[index];

        return (
            update.cid,
            update.submittedBy,
            update.timestamp
        );
    }
}