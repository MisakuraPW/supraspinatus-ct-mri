from __future__ import annotations

from pathlib import Path


def rigid_register_mri_label_to_ct(
    ct_image: str | Path,
    mri_image: str | Path,
    mri_label: str | Path,
    out_label: str | Path,
) -> None:
    """Map an MRI label into CT space using optional SimpleITK rigid registration.

    This is route D from the research notes. It is intentionally optional because
    the local prototype does not depend on SimpleITK, while cloud training
    machines usually can install it from `requirements/train.txt`.
    """

    try:
        import SimpleITK as sitk
    except Exception as exc:
        raise RuntimeError("SimpleITK is required for CT-MRI weak-label registration") from exc

    fixed = sitk.ReadImage(str(ct_image), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(mri_image), sitk.sitkFloat32)
    label = sitk.ReadImage(str(mri_label), sitk.sitkUInt8)

    init = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.1)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetInitialTransform(init, inPlace=False)
    transform = registration.Execute(fixed, moving)

    warped = sitk.Resample(label, fixed, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    Path(out_label).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(warped, str(out_label))

