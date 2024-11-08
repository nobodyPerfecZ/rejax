from rejax.algos import (
    DPPO,
    DQN,
    DSAC,
    IQN,
    PPO,
    PQN,
    SAC,
    TD3,
    Algorithm,
    DPPOKurt,
    DPPOSkew,
    PPOCVaRRejectionSampling,
)

_algos = {
    "crs_ppo": PPOCVaRRejectionSampling,
    "dppo": DPPO,
    "dppo_kurtosis": DPPOKurt,
    "dppo_skew": DPPOSkew,
    "dqn": DQN,
    "dsac": DSAC,
    "iqn": IQN,
    "ppo": PPO,
    "pqn": PQN,
    "sac": SAC,
    "td3": TD3,
}


def get_algo(algo: str) -> Algorithm:
    """Get an algorithm class."""
    return _algos[algo]


__all__ = [
    "get_algo",
    # Algorithms
    "DQN",
    "DSAC",
    "IQN",
    "PPO",
    "PPOCVaRRejectionSampling",
    "PQN",
    "SAC",
    "TD3",
    "DPPO",
    "DPPOKurt",
    "DPPOSkew",
]
