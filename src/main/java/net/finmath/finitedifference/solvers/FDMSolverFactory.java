package net.finmath.finitedifference.solvers;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBachelierModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMBatesModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMBlackScholesModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMCevModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FDMSabrModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.solvers.adi.BarrierPDEMode;
import net.finmath.finitedifference.solvers.adi.BarrierPreHitSpecification;
import net.finmath.finitedifference.solvers.adi.FDMBarrierHestonADI2D;
import net.finmath.finitedifference.solvers.adi.FDMBarrierSabrADI2D;
import net.finmath.finitedifference.solvers.adi.FDMBatesADI2D;
import net.finmath.finitedifference.solvers.adi.FDMHestonADI2D;
import net.finmath.finitedifference.solvers.adi.FDMSabrADI2D;
import net.finmath.modelling.Exercise;

/**
 * Centralized factory for choosing the finite-difference solver associated with
 * a given model / product / discretization / exercise combination.
 *
 * @author Alessandro Gnoatto
 */
public final class FDMSolverFactory {

	private FDMSolverFactory() {
		// Utility class
	}

	public static FDMSolver createSolver(
			final FiniteDifferenceEquityModel model,
			final FiniteDifferenceProduct product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final Exercise exercise) {

		if(isOneDimensionalJumpModel(model)) {
			return new FDMThetaMethod1DJump(
					model,
					product,
					spaceTimeDiscretization,
					exercise
			);
		}
		else if(model instanceof FDMBlackScholesModel
				|| model instanceof FDMCevModel
				|| model instanceof FDMBachelierModel) {
			return new FDMThetaMethod1D(
					model,
					product,
					spaceTimeDiscretization,
					exercise
			);
		}
		else if(model instanceof FDMBatesModel) {
			return new FDMBatesADI2D(
					(FDMBatesModel) model,
					product,
					spaceTimeDiscretization,
					exercise
			);
		}
		else if(model instanceof FDMHestonModel) {
			return new FDMHestonADI2D(
					(FDMHestonModel) model,
					product,
					spaceTimeDiscretization,
					exercise
			);
		}
		else if(model instanceof FDMSabrModel) {
			return new FDMSabrADI2D(
					(FDMSabrModel) model,
					product,
					spaceTimeDiscretization,
					exercise
			);
		}
		else {
			throw new IllegalArgumentException(
					"Unsupported model type: " + model.getClass().getName());
		}
	}

	public static FDMSolver createSolver(
			final FiniteDifferenceEquityModel model,
			final FiniteDifferenceProduct product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final Exercise exercise,
			final BarrierPDEMode barrierMode,
			final BarrierPreHitSpecification preHitSpecification) {

		if(isOneDimensionalJumpModel(model)) {
			return new FDMThetaMethod1DJump(
					model,
					product,
					spaceTimeDiscretization,
					exercise
			);
		}
		else if(model instanceof FDMBlackScholesModel
				|| model instanceof FDMCevModel
				|| model instanceof FDMBachelierModel) {
			return new FDMThetaMethod1D(
					model,
					product,
					spaceTimeDiscretization,
					exercise
			);
		}
		else if(model instanceof FDMBatesModel) {
			/*
			 * Barrier-specific Bates handling is not implemented yet.
			 * For the time being, ignore barrierMode and return the vanilla Bates solver.
			 */
			return new FDMBatesADI2D(
					(FDMBatesModel) model,
					product,
					spaceTimeDiscretization,
					exercise
			);
		}
		else if(model instanceof FDMHestonModel) {
			if(barrierMode == null) {
				return new FDMHestonADI2D(
						(FDMHestonModel) model,
						product,
						spaceTimeDiscretization,
						exercise
				);
			}

			return new FDMBarrierHestonADI2D(
					(FDMHestonModel) model,
					product,
					spaceTimeDiscretization,
					exercise,
					barrierMode,
					preHitSpecification
			);
		}
		else if(model instanceof FDMSabrModel) {
			if(barrierMode == null) {
				return new FDMSabrADI2D(
						(FDMSabrModel) model,
						product,
						spaceTimeDiscretization,
						exercise
				);
			}

			return new FDMBarrierSabrADI2D(
					(FDMSabrModel) model,
					product,
					spaceTimeDiscretization,
					exercise,
					barrierMode,
					preHitSpecification
			);
		}
		else {
			throw new IllegalArgumentException(
					"Unsupported model type: " + model.getClass().getName());
		}
	}

	public static FDMSolver createSolver(
			final FiniteDifferenceEquityModel model,
			final FiniteDifferenceProduct product,
			final Exercise exercise) {
		return createSolver(model, product, model.getSpaceTimeDiscretization(), exercise);
	}

	public static FDMSolver createSolver(
			final FiniteDifferenceEquityModel model,
			final FiniteDifferenceProduct product,
			final Exercise exercise,
			final BarrierPDEMode barrierMode,
			final BarrierPreHitSpecification preHitSpecification) {
		return createSolver(
				model,
				product,
				model.getSpaceTimeDiscretization(),
				exercise,
				barrierMode,
				preHitSpecification
		);
	}

	private static boolean isOneDimensionalJumpModel(final FiniteDifferenceEquityModel model) {
		return model.getJumpComponent().isPresent()
				&& model.getInitialValue() != null
				&& model.getInitialValue().length == 1;
	}
}