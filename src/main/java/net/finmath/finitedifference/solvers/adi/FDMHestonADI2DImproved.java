package net.finmath.finitedifference.solvers.adi;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.products.FiniteDifferenceProduct;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.modelling.Exercise;

/**
 * Improved vanilla 2D ADI solver for the Heston model.
 *
 * <p>
 * This solver is built on top of {@link AbstractSplitADI2D} and uses a
 * semidiscrete split provided by {@link HestonADI2DOperatorSplit}.
 * It is intended as the first clean Heston-specific ADI solver in the
 * improve-adi-schemes branch.
 * </p>
 *
 * <p>
 * The current implementation is aimed at vanilla and other non-barrier
 * products first. Barrier-specific enforcement should be introduced later
 * in dedicated subclasses, once the operator split and the ADI core are
 * validated on Heston Europeans.
 * </p>
 */
public class FDMHestonADI2DImproved extends AbstractSplitADI2D {

	private final FDMHestonModel hestonModel;

	public FDMHestonADI2DImproved(
			final FDMHestonModel model,
			final FiniteDifferenceProduct product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final Exercise exercise) {
		this(
				model,
				product,
				spaceTimeDiscretization,
				exercise,
				ADIScheme.DOUGLAS
		);
	}

	public FDMHestonADI2DImproved(
			final FDMHestonModel model,
			final FiniteDifferenceProduct product,
			final SpaceTimeDiscretization spaceTimeDiscretization,
			final Exercise exercise,
			final ADIScheme adiScheme) {
		super(
				model,
				product,
				spaceTimeDiscretization,
				exercise,
				new HestonADI2DOperatorSplit(
						model,
						spaceTimeDiscretization.getSpaceGrid(0).getGrid(),
						spaceTimeDiscretization.getSpaceGrid(1).getGrid()
				),
				adiScheme
		);
		this.hestonModel = model;
	}

	public FDMHestonModel getHestonModel() {
		return hestonModel;
	}
}