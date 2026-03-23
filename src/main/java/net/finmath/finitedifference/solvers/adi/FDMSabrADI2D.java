package net.finmath.finitedifference.solvers.adi;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMSabrModel;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.modelling.Exercise;

/**
 * ADI finite difference solver for two-dimensional SABR PDEs.
 *
 * <p>
 * This class specializes {@link AbstractADI2D} to the
 * {@link FDMSabrModel}. The generic ADI logic, line solves,
 * boundary enforcement, and exercise handling are inherited
 * from the abstract base class.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public class FDMSabrADI2D extends AbstractADI2D {

	public FDMSabrADI2D(
			final FDMSabrModel model,
			final FiniteDifferenceProduct product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final Exercise exercise) {
		super(model, product, spaceTimeDiscretization, exercise);
	}
}