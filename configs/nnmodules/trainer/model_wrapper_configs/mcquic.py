config = dict(
    # TODO: use Lamb optimizer?
    optimizer_type="Adam",
    optimizer_config=dict(
        base_lr=1e-4,
    ),
    scheduler_type="CosineAnnealingLR",
    scheduler_config=dict(
        T_max=1000,
        eta_min=1e-6,
    ),
)