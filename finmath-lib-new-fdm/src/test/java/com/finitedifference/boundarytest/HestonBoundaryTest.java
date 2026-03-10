package com.finitedifference.boundarytest;

import java.time.LocalDate;

import org.junit.Assert;
import org.junit.Test;

import net.finmath.finitedifference.assetderivativevaluation.models.FDMHestonModel;
import net.finmath.finitedifference.assetderivativevaluation.models.FiniteDifferenceEquityModel;
import net.finmath.finitedifference.assetderivativevaluation.products.EuropeanOption;
import net.finmath.finitedifference.grids.Grid;
import net.finmath.finitedifference.grids.SpaceTimeDiscretization;
import net.finmath.finitedifference.grids.UniformGrid;
import net.finmath.marketdata.model.curves.CurveInterpolation.ExtrapolationMethod;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationEntity;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationMethod;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterpolation;
import net.finmath.modelling.products.CallOrPut;
import net.finmath.time.TimeDiscretization;
import net.finmath.time.TimeDiscretizationFromArray;

public class HestonBoundaryTest {

    private static DiscountCurve flatCurve(final String name, final double rate) {
        final double[] times = new double[] {0.0, 1.0, 2.0, 5.0};
        final double[] zeros = new double[] {rate, rate, rate, rate};
        return DiscountCurveInterpolation.createDiscountCurveFromAnnualizedZeroRates(
                name, LocalDate.of(2010, 8, 1),
                times, zeros,
                InterpolationMethod.LINEAR,
                ExtrapolationMethod.CONSTANT,
                InterpolationEntity.LOG_OF_VALUE_PER_TIME
        );
    }

    private static FiniteDifferenceEquityModel createModel(final double r, final double q, final double maturity) {
        final double s0 = 100.0;
        final double v0 = 0.04;

        final DiscountCurve rf = flatCurve("rf", r);
        final DiscountCurve dq = flatCurve("dq", q);

        final Grid sGrid = new UniformGrid(20, 0.0, 200.0);
        final Grid vGrid = new UniformGrid(20, 0.0, 1.0);

        final TimeDiscretization timeDisc = new TimeDiscretizationFromArray(0.0, 10, maturity/10.0);

        final SpaceTimeDiscretization disc = new SpaceTimeDiscretization(new Grid[] {sGrid, vGrid}, timeDisc, 0.5, new double[] {s0, v0});

        // Heston parameters (not used by boundary tests except stored)
        final double kappa = 2.0;
        final double thetaV = 0.04;
        final double sigma = 0.5;
        final double rho = -0.7;

        return new FDMHestonModel(s0, v0, rf, dq, kappa, thetaV, sigma, rho, disc);
    }

    @Test
    public void testEuropeanCallBoundaries() {
        final double maturity = 1.0;
        final double r = 0.05;
        final double q = 0.02;

        final FiniteDifferenceEquityModel model = createModel(r, q, maturity);
        final double strike = 100.0;

        final EuropeanOption option = new EuropeanOption(maturity, strike, CallOrPut.CALL);
        
        final double time = 0.25;
        final double S = 120.0;
        final double v = 0.09;

        final double[] lower = model.getValueAtLowerBoundary(option, time, S, v);
        final double[] upper = model.getValueAtUpperBoundary(option, time, S, v);

        final double discountR = Math.exp(-r * (maturity - time));
        final double discountQ = Math.exp(-q * (maturity - time));

        // Lower: S->0 is 0 for call; v->0 is intrinsic of forward-like value
        Assert.assertEquals(0.0, lower[0], 1e-0);
        Assert.assertEquals(Math.max(S * discountQ - strike * discountR, 0.0), lower[1], 1e-0);

        // Upper: S->inf approx linear payoff; v->inf tends to S*discountQ
        Assert.assertEquals(S * discountQ - strike * discountR, upper[0], 1e-0);
        Assert.assertEquals(S * discountQ, upper[1], 1e-0);
    }

    @Test
    public void testEuropeanPutBoundaries() {
        final double maturity = 1.0;
        final double r = 0.03;
        final double q = 0.01;

        final FiniteDifferenceEquityModel model = createModel(r, q, maturity);
        final double strike = 110.0;

        final EuropeanOption option = new EuropeanOption(maturity, strike, CallOrPut.PUT);

        final double time = 0.6;
        final double S = 90.0;
        final double v = 0.16;

        final double[] lower = model.getValueAtLowerBoundary(option, time, S, v);
        final double[] upper = model.getValueAtUpperBoundary(option, time, S, v);

        final double discountR = Math.exp(-r * (maturity - time));
        final double discountQ = Math.exp(-q * (maturity - time));

        // Lower: S->0 is discounted strike for put; v->0 is intrinsic
        Assert.assertEquals(strike * discountR, lower[0], 1e-0);
        Assert.assertEquals(Math.max(strike * discountR - S * discountQ, 0.0), lower[1], 1e-0);

        // Upper: S->inf is 0 for put; v->inf tends to discounted strike
        Assert.assertEquals(0.0, upper[0], 1e-0);
        Assert.assertEquals(strike * discountR, upper[1], 1e-0);
    }
}
