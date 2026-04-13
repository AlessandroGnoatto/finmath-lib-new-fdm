package it.univr.fima.correction;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.Collection;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import it.univr.fima.correction.BarrierOptions.BarrierType;
import it.univr.fima.correction.BarrierOptions.BinaryBarrierEventType;
import it.univr.fima.correction.BarrierOptions.BinaryPayoffType;

/**
 * Regression test matrix for the 28 single-barrier binary products listed in
 * Haug, Table 4-22.
 *
 * Inputs from the table:
 * H = 100, T = 0.5, r = 0.1, b = 0.1, sigma = 0.2
 * cash payoff K = 15, except products (3) and (4), where the asset-at-hit payoff is H.
 *
 * Since b = r - q and the table uses b = r = 0.1, we set q = 0.0.
 *
 * Note:
 * Some printed entries in the table appear inconsistent with the formulas listed
 * on the same pages. This test uses formula-consistent values for those rows.
 */
@RunWith(Parameterized.class)
public class BarrierOptionsHaugBinaryBarrierTableTest {

    private static final double H = 100.0;
    private static final double T = 0.5;
    private static final double R = 0.1;
    private static final double Q = 0.0;
    private static final double SIGMA = 0.2;
    private static final double CASH = 15.0;

    private static final double TOL = 2E-4;

    @Parameterized.Parameter(0)
    public int productNumber;

    @Parameterized.Parameter(1)
    public double spot;

    @Parameterized.Parameter(2)
    public double strike;

    @Parameterized.Parameter(3)
    public double expectedValue;

    @Parameterized.Parameters(name = "product={0}, S={1}, X={2}")
    public static Collection<Object[]> data() {
        return Arrays.asList(new Object[][] {
                {1, 105.0, 102.0, 9.7264},
                {1, 105.0, 98.0,  9.7264},

                {2, 95.0, 102.0, 11.6553},
                {2, 95.0, 98.0,  11.6553},

                // corrected
                {3, 105.0, 102.0, 64.8426},
                {3, 105.0, 98.0,  64.8426},

                // corrected
                {4, 95.0, 102.0, 77.7017},
                {4, 95.0, 98.0,  77.7017},

                {5, 105.0, 102.0, 9.3604},
                {5, 105.0, 98.0,  9.3604},

                {6, 95.0, 102.0, 11.2223},
                {6, 95.0, 98.0,  11.2223},

                {7, 105.0, 102.0, 64.8426},
                {7, 105.0, 98.0,  64.8426},

                {8, 95.0, 102.0, 77.7017},
                {8, 95.0, 98.0,  77.7017},

                {9, 105.0, 102.0, 4.9081},
                {9, 105.0, 98.0,  4.9081},

                {10, 95.0, 102.0, 3.0461},
                {10, 95.0, 98.0,  3.0461},

                {11, 105.0, 102.0, 40.1574},
                {11, 105.0, 98.0,  40.1574},

                {12, 95.0, 102.0, 17.2983},
                {12, 95.0, 98.0,  17.2983},

                {13, 105.0, 102.0, 4.9289},
                {13, 105.0, 98.0,  6.2150},

                // corrected for X > H
                {14, 95.0, 102.0, 5.8926},
                {14, 95.0, 98.0,  7.4519},

                {15, 105.0, 102.0, 37.2782},
                {15, 105.0, 98.0,  45.8530},

                {16, 95.0, 102.0, 44.5294},
                {16, 95.0, 98.0,  54.9262},

                {17, 105.0, 102.0, 4.4314},
                {17, 105.0, 98.0,  3.1454},

                {18, 95.0, 102.0, 5.3297},
                {18, 95.0, 98.0,  3.7704},

                {19, 105.0, 102.0, 27.5644},
                {19, 105.0, 98.0,  18.9896},

                // corrected for X > H
                {20, 95.0, 102.0, 33.1723},
                {20, 95.0, 98.0,  22.7755},

                {21, 105.0, 102.0, 4.8758},
                {21, 105.0, 98.0,  4.9081},

                {22, 95.0, 102.0, 0.0000},
                {22, 95.0, 98.0,  0.0407},

                {23, 105.0, 102.0, 39.9391},
                {23, 105.0, 98.0,  40.1574},

                {24, 95.0, 102.0, 0.0000},
                {24, 95.0, 98.0,  0.2676},

                {25, 105.0, 102.0, 0.0323},
                {25, 105.0, 98.0,  0.0000},

                {26, 95.0, 102.0, 3.0461},
                {26, 95.0, 98.0,  3.0054},

                {27, 105.0, 102.0, 0.2183},
                {27, 105.0, 98.0,  0.0000},

                {28, 95.0, 102.0, 17.2983},
                {28, 95.0, 98.0,  17.0306}
        });
    }

    @Test
    public void testHaugBinaryBarrierTable() {
        final double value = valueForProduct(productNumber, spot, strike);

        assertEquals(
                "Mismatch for product " + productNumber + " with S=" + spot + ", X=" + strike,
                expectedValue,
                value,
                TOL);
    }

    private static double valueForProduct(
            final int productNumber,
            final double spot,
            final double strike) {

        switch (productNumber) {
        case 1:
            return BarrierOptions.blackScholesBinaryBarrierAtHitValue(
                    spot, R, Q, SIGMA, T, H,
                    BarrierType.DOWN_IN,
                    BinaryPayoffType.CASH_OR_NOTHING,
                    CASH);

        case 2:
            return BarrierOptions.blackScholesBinaryBarrierAtHitValue(
                    spot, R, Q, SIGMA, T, H,
                    BarrierType.UP_IN,
                    BinaryPayoffType.CASH_OR_NOTHING,
                    CASH);

        case 3:
            return BarrierOptions.blackScholesBinaryBarrierAtHitValue(
                    spot, R, Q, SIGMA, T, H,
                    BarrierType.DOWN_IN,
                    BinaryPayoffType.ASSET_OR_NOTHING,
                    0.0);

        case 4:
            return BarrierOptions.blackScholesBinaryBarrierAtHitValue(
                    spot, R, Q, SIGMA, T, H,
                    BarrierType.UP_IN,
                    BinaryPayoffType.ASSET_OR_NOTHING,
                    0.0);

        case 5:
            return BarrierOptions.blackScholesBinaryBarrierStatusAtExpiryValue(
                    spot, R, Q, SIGMA, T, H,
                    BarrierType.DOWN_IN,
                    BinaryBarrierEventType.HIT,
                    BinaryPayoffType.CASH_OR_NOTHING,
                    CASH);

        case 6:
            return BarrierOptions.blackScholesBinaryBarrierStatusAtExpiryValue(
                    spot, R, Q, SIGMA, T, H,
                    BarrierType.UP_IN,
                    BinaryBarrierEventType.HIT,
                    BinaryPayoffType.CASH_OR_NOTHING,
                    CASH);

        case 7:
            return BarrierOptions.blackScholesBinaryBarrierStatusAtExpiryValue(
                    spot, R, Q, SIGMA, T, H,
                    BarrierType.DOWN_IN,
                    BinaryBarrierEventType.HIT,
                    BinaryPayoffType.ASSET_OR_NOTHING,
                    0.0);

        case 8:
            return BarrierOptions.blackScholesBinaryBarrierStatusAtExpiryValue(
                    spot, R, Q, SIGMA, T, H,
                    BarrierType.UP_IN,
                    BinaryBarrierEventType.HIT,
                    BinaryPayoffType.ASSET_OR_NOTHING,
                    0.0);

        case 9:
            return BarrierOptions.blackScholesBinaryBarrierStatusAtExpiryValue(
                    spot, R, Q, SIGMA, T, H,
                    BarrierType.DOWN_OUT,
                    BinaryBarrierEventType.NO_HIT,
                    BinaryPayoffType.CASH_OR_NOTHING,
                    CASH);

        case 10:
            return BarrierOptions.blackScholesBinaryBarrierStatusAtExpiryValue(
                    spot, R, Q, SIGMA, T, H,
                    BarrierType.UP_OUT,
                    BinaryBarrierEventType.NO_HIT,
                    BinaryPayoffType.CASH_OR_NOTHING,
                    CASH);

        case 11:
            return BarrierOptions.blackScholesBinaryBarrierStatusAtExpiryValue(
                    spot, R, Q, SIGMA, T, H,
                    BarrierType.DOWN_OUT,
                    BinaryBarrierEventType.NO_HIT,
                    BinaryPayoffType.ASSET_OR_NOTHING,
                    0.0);

        case 12:
            return BarrierOptions.blackScholesBinaryBarrierStatusAtExpiryValue(
                    spot, R, Q, SIGMA, T, H,
                    BarrierType.UP_OUT,
                    BinaryBarrierEventType.NO_HIT,
                    BinaryPayoffType.ASSET_OR_NOTHING,
                    0.0);

        case 13:
            return BarrierOptions.blackScholesBinaryBarrierOptionValue(
                    spot, R, Q, SIGMA, T, strike, true, H,
                    BarrierType.DOWN_IN,
                    BinaryPayoffType.CASH_OR_NOTHING,
                    CASH);

        case 14:
            return BarrierOptions.blackScholesBinaryBarrierOptionValue(
                    spot, R, Q, SIGMA, T, strike, true, H,
                    BarrierType.UP_IN,
                    BinaryPayoffType.CASH_OR_NOTHING,
                    CASH);

        case 15:
            return BarrierOptions.blackScholesBinaryBarrierOptionValue(
                    spot, R, Q, SIGMA, T, strike, true, H,
                    BarrierType.DOWN_IN,
                    BinaryPayoffType.ASSET_OR_NOTHING,
                    0.0);

        case 16:
            return BarrierOptions.blackScholesBinaryBarrierOptionValue(
                    spot, R, Q, SIGMA, T, strike, true, H,
                    BarrierType.UP_IN,
                    BinaryPayoffType.ASSET_OR_NOTHING,
                    0.0);

        case 17:
            return BarrierOptions.blackScholesBinaryBarrierOptionValue(
                    spot, R, Q, SIGMA, T, strike, false, H,
                    BarrierType.DOWN_IN,
                    BinaryPayoffType.CASH_OR_NOTHING,
                    CASH);

        case 18:
            return BarrierOptions.blackScholesBinaryBarrierOptionValue(
                    spot, R, Q, SIGMA, T, strike, false, H,
                    BarrierType.UP_IN,
                    BinaryPayoffType.CASH_OR_NOTHING,
                    CASH);

        case 19:
            return BarrierOptions.blackScholesBinaryBarrierOptionValue(
                    spot, R, Q, SIGMA, T, strike, false, H,
                    BarrierType.DOWN_IN,
                    BinaryPayoffType.ASSET_OR_NOTHING,
                    0.0);

        case 20:
            return BarrierOptions.blackScholesBinaryBarrierOptionValue(
                    spot, R, Q, SIGMA, T, strike, false, H,
                    BarrierType.UP_IN,
                    BinaryPayoffType.ASSET_OR_NOTHING,
                    0.0);

        case 21:
            return BarrierOptions.blackScholesBinaryBarrierOptionValue(
                    spot, R, Q, SIGMA, T, strike, true, H,
                    BarrierType.DOWN_OUT,
                    BinaryPayoffType.CASH_OR_NOTHING,
                    CASH);

        case 22:
            return BarrierOptions.blackScholesBinaryBarrierOptionValue(
                    spot, R, Q, SIGMA, T, strike, true, H,
                    BarrierType.UP_OUT,
                    BinaryPayoffType.CASH_OR_NOTHING,
                    CASH);

        case 23:
            return BarrierOptions.blackScholesBinaryBarrierOptionValue(
                    spot, R, Q, SIGMA, T, strike, true, H,
                    BarrierType.DOWN_OUT,
                    BinaryPayoffType.ASSET_OR_NOTHING,
                    0.0);

        case 24:
            return BarrierOptions.blackScholesBinaryBarrierOptionValue(
                    spot, R, Q, SIGMA, T, strike, true, H,
                    BarrierType.UP_OUT,
                    BinaryPayoffType.ASSET_OR_NOTHING,
                    0.0);

        case 25:
            return BarrierOptions.blackScholesBinaryBarrierOptionValue(
                    spot, R, Q, SIGMA, T, strike, false, H,
                    BarrierType.DOWN_OUT,
                    BinaryPayoffType.CASH_OR_NOTHING,
                    CASH);

        case 26:
            return BarrierOptions.blackScholesBinaryBarrierOptionValue(
                    spot, R, Q, SIGMA, T, strike, false, H,
                    BarrierType.UP_OUT,
                    BinaryPayoffType.CASH_OR_NOTHING,
                    CASH);

        case 27:
            return BarrierOptions.blackScholesBinaryBarrierOptionValue(
                    spot, R, Q, SIGMA, T, strike, false, H,
                    BarrierType.DOWN_OUT,
                    BinaryPayoffType.ASSET_OR_NOTHING,
                    0.0);

        case 28:
            return BarrierOptions.blackScholesBinaryBarrierOptionValue(
                    spot, R, Q, SIGMA, T, strike, false, H,
                    BarrierType.UP_OUT,
                    BinaryPayoffType.ASSET_OR_NOTHING,
                    0.0);

        default:
            throw new IllegalArgumentException("Unsupported product number: " + productNumber);
        }
    }
}