config = dict(
    # TODO: use Lamb optimizer?
    optimizer_type="Adam",
    optimizer_config=dict(
        base_lr=2e-4,
    ),
    scheduler_type="CosineAnnealingLR",
    scheduler_config=dict(
        T_max=1000,
        eta_min=2e-6,
    ),
)