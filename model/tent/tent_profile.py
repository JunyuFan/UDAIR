def _make_tta_profile(dataset_name, source_tasks, steps=5, image_size=256, feature_bank='source_domain_distrib.pth'):
    catalog = {name: idx for idx, name in enumerate(source_tasks)}
    profile = {
        'enabled': dataset_name not in catalog,
        'steps': steps,
        'feature_bank': feature_bank,
        'image_size': image_size,
    }
    if not profile['enabled']:
        return profile

    for name, index in catalog.items():
        if dataset_name.startswith(f'{name}_'):
            profile['reference_index'] = index
            return profile
    raise ValueError(f"Unknown target-domain dataset: {dataset_name}")