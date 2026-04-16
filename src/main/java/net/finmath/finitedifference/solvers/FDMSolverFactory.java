package net.finmath.finitedifference.solvers;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMBachelierModel;
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
import net.finmath.finitedifference.solvers.adi.FDMHestonADI2D;
import net.finmath.finitedifference.solvers.adi.FDMSabrADI2D;
import net.finmath.modelling.Exercise;

/**
 * Centralized factory for choosing the finite-difference solver associated with
 * a given model / product / discretization / exercise combination.
 *
 * <p>
 * The supplied {@link SpaceTimeDiscretization} is used directly for all
 * supported solver paths, including the 2D ADI solvers for Heston and SABR.
 * This makes product-level discretization refinement effective in 2D as well.
 * </p>
 *
 * <p>
 * A barrier-aware overload is also provided for the direct 2D knock-in
 * pre-hit / interface formulation.
 * </p>
 *
 * @author Alessandro Gnoatto
 */
public final class FDMSolverFactory {

    private FDMSolverFactory() {
        // Utility class
    }

    /**
     * Standard factory overload.
     *
     * @param model The finite-difference model.
     * @param product The product to be priced.
     * @param spaceTimeDiscretization The discretization to use.
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
            return new FDMThetaMethod1D(
                    model,
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
     * Barrier-aware factory overload used by the direct 2D knock-in path.
     *
     * <p>
     * For 1D models, barrierMode / preHitSpecification are ignored and the
     * standard 1D theta solver is returned.
     * </p>
     *
     * @param model The finite-difference model.
     * @param product The product to be priced.
     * @param spaceTimeDiscretization The discretization to use.
     * @param exercise The exercise specification.
     * @param barrierMode The barrier PDE mode. May be null for ordinary pricing.
     * @param preHitSpecification The pre-hit specification for direct knock-in
     *        interface pricing. May be null when not required.
     * @return The corresponding solver.
     */
    public static FDMSolver createSolver(
            final FiniteDifferenceEquityModel model,
            final FiniteDifferenceProduct product,
            final SpaceTimeDiscretization spaceTimeDiscretization,
            final Exercise exercise,
            final BarrierPDEMode barrierMode,
            final BarrierPreHitSpecification preHitSpecification) {

        if(model instanceof FDMBlackScholesModel
                || model instanceof FDMCevModel
                || model instanceof FDMBachelierModel) {
            return new FDMThetaMethod1D(
                    model,
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

    /**
     * Barrier-aware convenience overload using the model's own discretization.
     *
     * @param model The finite-difference model.
     * @param product The product to be priced.
     * @param exercise The exercise specification.
     * @param barrierMode The barrier PDE mode.
     * @param preHitSpecification The pre-hit specification.
     * @return The corresponding solver.
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
}