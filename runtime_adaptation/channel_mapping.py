"""Explicit source-channel policies for standard-64 FBMAE adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Mapping

from runtime_adaptation.contracts import CanonicalEEGWindow


@dataclass(frozen=True)
class ChannelMappingPolicy:
    name: str
    dataset_key: str
    channel_map_id: str
    allowed_dropped_channels: FrozenSet[str]


CHANNEL_MAPPING_POLICIES: Mapping[str, ChannelMappingPolicy] = {
    "cf8399da04dbb6bb": ChannelMappingPolicy(
        name="inria-exact-standard64-names",
        dataset_key="inria",
        channel_map_id="cf8399da04dbb6bb",
        allowed_dropped_channels=frozenset(),
    ),
    "85f920b8a8464acd": ChannelMappingPolicy(
        name="egobrain-drop-4-unmatched",
        dataset_key="egobrain",
        channel_map_id="85f920b8a8464acd",
        allowed_dropped_channels=frozenset({"FT9", "PO9", "PO10", "FT10"}),
    ),
    "633f69edb30cb395": ChannelMappingPolicy(
        name="shin2017a-direct-name-validation-only",
        dataset_key="shin2017a",
        channel_map_id="633f69edb30cb395",
        allowed_dropped_channels=frozenset(
            {
                "AFF5h", "AFp1", "AFp2", "AFF6h", "AFF1h", "AFF2h",
                "FCC5h", "FCC3h", "CCP5h", "CCP3h", "PPO1h", "POO1",
                "POO2", "PPO2h", "FCC4h", "FCC6h", "CCP4h", "CCP6h",
            }
        ),
    ),
}


def resolve_channel_mapping_policy(window: CanonicalEEGWindow) -> ChannelMappingPolicy:
    map_id = str(window.metadata.get("channel_map_id", ""))
    if map_id:
        policy = CHANNEL_MAPPING_POLICIES.get(map_id)
        if policy is None:
            raise ValueError(
                f"No explicit standard-64 policy for channel_map_id={map_id!r} "
                f"({window.dataset_key})"
            )
        if policy.dataset_key != window.dataset_key:
            raise ValueError(
                f"Channel-map policy {policy.name!r} belongs to "
                f"{policy.dataset_key!r}, not {window.dataset_key!r}"
            )
        return policy

    # Backward-compatible strict mode for manually constructed windows.
    return ChannelMappingPolicy(
        name="strict-no-drop",
        dataset_key=window.dataset_key,
        channel_map_id="",
        allowed_dropped_channels=frozenset(),
    )
