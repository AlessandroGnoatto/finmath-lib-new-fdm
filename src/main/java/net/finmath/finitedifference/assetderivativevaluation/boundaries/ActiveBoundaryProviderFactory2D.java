package net.finmath.finitedifference.assetderivativevaluation.boundaries;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.solvers.TwoStateActiveBoundaryProvider2D;
import net.finmath.modelling.products.CallOrPut;

/**
 * Factory for 2D active-state boundary providers used by the direct two-state knock-in solver.
 */
public final class ActiveBoundaryProviderFactory2D {

	private ActiveBoundaryProviderFactory2D() {
	}

	public static TwoStateActiveBoundaryProvider2D createProvider(
			final Object model,
			final double strike,
			final double maturity,
			final CallOrPut callOrPut) {

		if(model instanceof FDMHestonModel) {
			return new HestonActiveBoundaryProvider(
					(FDMHestonModel)model,
					strike,
					maturity,
					callOrPut
			);
		}

		throw new IllegalArgumentException(
				"No 2D active boundary provider available for model type: " + model.getClass().getName());
	}
}