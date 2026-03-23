package net.finmath.finitedifference.solvers;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBachelierModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMCevModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMSabrModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.solvers.adi.FDMHestonADI2D;
import net.finmath.finitedifference.solvers.adi.FDMSabrADI2D;
import net.finmath.modelling.Exercise;

/**
 * Centralized resolver for choosing the finite-difference solver associated with
 * a given model / product / discretization / exercise combination.
 *
 * <p>
 * This class is intentionally small and conservative:
 * it centralizes the existing instanceof-based dispatch without changing the
 * public product API ({@code product.getValue(model)}).
 * </p>
 *
 * <p>
 * The supplied {@link SpaceTimeDiscretization} is used directly for 1D theta-method
 * solvers. For current 2D ADI solvers, the model's own discretization is used,
 * preserving the existing behavior in the product classes.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public final class FDMSolverFactory {

	private FDMSolverFactory() {
		// Utility class
	}

	/**
	 * Creates the appropriate finite-difference solver for the given model.
	 *
	 * @param model The finite-difference model.
	 * @param product The product to be priced.
	 * @param spaceTimeDiscretization The discretization to use where supported.
	 * @param exercise The exercise specification.
	 * @return The corresponding solver.
	 */
	public static FDMSolver createSolver(
			final FiniteDifferenceEquityModel model,
			final FiniteDifferenceProduct product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final Exercise exercise) {

		if(model instanceof FDMBlackScholesModel
				|| model instanceof FDMCevModel
				|| model instanceof FDMBachelierModel) {
			return new FDMThetaMethod1D(model, product, spaceTimeDiscretization, exercise);
		}
		else if(model instanceof FDMHestonModel) {
			/*
			 * Preserve current behavior:
			 * existing products use model.getSpaceTimeDiscretization() for Heston ADI.
			 */
			return new FDMHestonADI2D(
					(FDMHestonModel) model,
					product,
					model.getSpaceTimeDiscretization(),
					exercise
			);
		}
		else if(model instanceof FDMSabrModel) {
			/*
			 * Preserve current behavior:
			 * existing products use model.getSpaceTimeDiscretization() for SABR ADI.
			 */
			return new FDMSabrADI2D(
					(FDMSabrModel) model,
					product,
					model.getSpaceTimeDiscretization(),
					exercise
			);
		}
		else {
			throw new IllegalArgumentException(
					"Unsupported model type: " + model.getClass().getName());
		}
	}

	/**
	 * Convenience overload using the model's own discretization.
	 *
	 * @param model The finite-difference model.
	 * @param product The product to be priced.
	 * @param exercise The exercise specification.
	 * @return The corresponding solver.
	 */
	public static FDMSolver createSolver(
			final FiniteDifferenceEquityModel model,
			final FiniteDifferenceProduct product,
			final Exercise exercise) {
		return createSolver(model, product, model.getSpaceTimeDiscretization(), exercise);
	}
}