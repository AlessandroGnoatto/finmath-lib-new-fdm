package net.finmath.marketdata.model.volatilities;

import static org.junit.Assert.assertEquals;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

import net.finmath.marketdata.model.curves.CurveInterpolation.ExtrapolationMethod;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationEntity;
import net.finmath.marketdata.model.curves.CurveInterpolation.InterpolationMethod;
import net.finmath.marketdata.model.curves.DiscountCurve;
import net.finmath.marketdata.model.curves.DiscountCurveInterpolation;
import net.finmath.marketdata.model.volatilities.VolatilitySurface.QuotingConvention;

/**
 * Tests for {@link DupireLocalVolatility}.
 *
 * @author Alessandro Gnoatto
 */
public class DupireLocalVolatilityTest {

	private static final double TOLERANCE = 1E-8;

	@Test
	public void testFlatImpliedVolatilitySurfaceGivesFlatLocalVolatilitySurface() {

		final double spot = 100.0;
		final double riskFreeRate = 0.05;
		final double dividendYield = 0.02;
		final double flatVolatility = 0.20;

		final LocalDate referenceDate = LocalDate.of(2020, 1, 1);

		final double[] maturities = new double[] {
				0.25,
				0.50,
				1.00,
				2.00
		};

		final double[] strikes = new double[] {
				70.0,
				80.0,
				90.0,
				100.0,
				110.0,
				120.0,
				130.0
		};

		final List<OptionData> optionQuotes = new ArrayList<>();

		for(final double maturity : maturities) {
			for(final double strike : strikes) {
				optionQuotes.add(
						new OptionData(
								"underlying",
								referenceDate,
								strike,
								maturity,
								flatVolatility,
								QuotingConvention.VOLATILITYLOGNORMAL
						)
				);
			}
		}

		final DiscountCurve discountCurve = createDiscountCurve(
				"discountCurve",
				riskFreeRate,
				new double[] { 0.0, 0.25, 0.50, 1.00, 2.00, 3.00 }
		);

		final DiscountCurve equityForwardCurve = createEquityForwardCurve(
				"equityForwardCurve",
				spot,
				riskFreeRate,
				dividendYield,
				new double[] { 0.0, 0.25, 0.50, 1.00, 2.00, 3.00 }
		);

		final OptionSurfaceDataInterpolated impliedVolatilitySurface =
				OptionSurfaceDataInterpolated.ofUnsorted(
						optionQuotes.toArray(new OptionData[0]),
						discountCurve,
						equityForwardCurve
				);

		final LocalVolatility localVolatility =
				new DupireLocalVolatility(impliedVolatilitySurface);

		assertEquals(flatVolatility, localVolatility.getValue(0.25, 80.0), TOLERANCE);
		assertEquals(flatVolatility, localVolatility.getValue(0.50, 100.0), TOLERANCE);
		assertEquals(flatVolatility, localVolatility.getValue(1.00, 120.0), TOLERANCE);
		assertEquals(flatVolatility, localVolatility.getValue(1.50, 95.0), TOLERANCE);
		assertEquals(flatVolatility, localVolatility.getValue(2.00, 110.0), TOLERANCE);
	}

	private static DiscountCurve createDiscountCurve(
			final String name,
			final double riskFreeRate,
			final double[] times) {

		final double[] discountFactors = new double[times.length];

		for(int timeIndex = 0; timeIndex < times.length; timeIndex++) {
			discountFactors[timeIndex] = Math.exp(-riskFreeRate * times[timeIndex]);
		}

		return DiscountCurveInterpolation.createDiscountCurveFromDiscountFactors(
				name,
				times,
				discountFactors,
				InterpolationMethod.LINEAR,
				ExtrapolationMethod.CONSTANT,
				InterpolationEntity.VALUE
		);
	}

	private static DiscountCurve createEquityForwardCurve(
			final String name,
			final double spot,
			final double riskFreeRate,
			final double dividendYield,
			final double[] times) {

		final double[] forwardValues = new double[times.length];

		for(int timeIndex = 0; timeIndex < times.length; timeIndex++) {
			forwardValues[timeIndex] =
					spot * Math.exp((riskFreeRate - dividendYield) * times[timeIndex]);
		}

		return DiscountCurveInterpolation.createDiscountCurveFromDiscountFactors(
				name,
				times,
				forwardValues,
				InterpolationMethod.LINEAR,
				ExtrapolationMethod.CONSTANT,
				InterpolationEntity.VALUE
		);
	}
}