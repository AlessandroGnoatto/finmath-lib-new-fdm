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
 * Centralized factory for selecting the finite-difference solver associated with a given
 * model, product, discretization, and exercise specification.
 *
 * <p>
 * The purpose of this class is to map a model class to the solver implementation capable
 * of handling the corresponding PDE or PIDE. In abstract terms, if the model leads to a
 * pricing equation of the form
 * </p>
 *
 * <p>
 * <i>
 * \frac{\partial V}{\partial t} + \mathcal{L}V = 0
 * </i>
 * </p>
 *
 * <p>
 * with generator <i>\mathcal{L}</i>, then this factory selects the discretization engine
 * used to approximate the operator <i>\mathcal{L}</i> on the supplied
 * {@link SpaceTimeDiscretization}. For one-dimensional diffusion models this is typically
 * a theta-method solver, while for two-dimensional stochastic-volatility or jump-diffusion
 * models this is typically an ADI-based solver.
 * </p>
 *
 * <p>
 * The current selection logic is:
 * </p>
 * <ul>
 *   <li>one-dimensional jump models &rarr; {@link FDMThetaMethod1DJump},</li>
 *   <li>Black-Scholes, CEV, and Bachelier models &rarr; {@link FDMThetaMethod1D},</li>
 *   <li>Bates models &rarr; {@link FDMBatesADI2D},</li>
 *   <li>Heston models &rarr; {@link FDMHestonADI2D} or
 *       {@link FDMBarrierHestonADI2D},</li>
 *   <li>SABR models &rarr; {@link FDMSabrADI2D} or
 *       {@link FDMBarrierSabrADI2D}.</li>
 * </ul>
 *
 * <p>
 * For barrier-aware two-dimensional Heston and SABR problems, the factory may select a
 * specialized pre-hit / barrier solver depending on the supplied
 * {@link BarrierPDEMode} and {@link BarrierPreHitSpecification}. If no barrier mode is
 * provided, the corresponding vanilla solver is returned.
 * </p>
 *
 * <p>
 * The class is a pure utility holder and cannot be instantiated.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public final class FDMSolverFactory {

	private FDMSolverFactory() {
		// Utility class
	}

	/**
	 * Creates a solver for the given model, product, discretization, and exercise.
	 *
	 * @param model The finite-difference model.
	 * @param product The product to be valued.
	 * @param spaceTimeDiscretization The space-time discretization.
	 * @param exercise The exercise specification.
	 * @return The selected finite-difference solver.
	 */
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

	/**
	 * Creates a solver for the given model, product, discretization, exercise, and barrier
	 * PDE configuration.
	 *
	 * @param model The finite-difference model.
	 * @param product The product to be valued.
	 * @param spaceTimeDiscretization The space-time discretization.
	 * @param exercise The exercise specification.
	 * @param barrierMode The barrier PDE mode.
	 * @param preHitSpecification The pre-hit barrier specification.
	 * @return The selected finite-difference solver.
	 */
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

	/**
	 * Creates a solver using the model's own space-time discretization.
	 *
	 * @param model The finite-difference model.
	 * @param product The product to be valued.
	 * @param exercise The exercise specification.
	 * @return The selected finite-difference solver.
	 */
	public static FDMSolver createSolver(
			final FiniteDifferenceEquityModel model,
			final FiniteDifferenceProduct product,
			final Exercise exercise) {
		return createSolver(model, product, model.getSpaceTimeDiscretization(), exercise);
	}

	/**
	 * Creates a barrier-aware solver using the model's own space-time discretization.
	 *
	 * @param model The finite-difference model.
	 * @param product The product to be valued.
	 * @param exercise The exercise specification.
	 * @param barrierMode The barrier PDE mode.
	 * @param preHitSpecification The pre-hit barrier specification.
	 * @return The selected finite-difference solver.
	 */
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