# # physics_engine.py
# import numpy as np
# import config
#
#
# class RefineryPhysics:
#     @staticmethod
#     def calculate_moist_density(temp_c, rh):
#         """
#         Calculates density of moist air.
#         Refineries are often hot, reducing air density and turbine efficiency.
#         """
#         # Convert Temp to Kelvin
#         Tk = temp_c + 273.15
#
#         # Saturation Vapor Pressure (Tetens Formula)
#         es = 6.1078 * 10 ** ((7.5 * temp_c) / (temp_c + 237.3))
#
#         # Vapor Pressure
#         pv = es * (rh / 100.0)
#
#         # Dry Air Pressure
#         pdry = config.STD_PRESSURE - pv
#
#         # Density Calculation (kg/m^3)
#         rho = (pdry * 100) / (config.RD_GAS_CONSTANT * Tk) + \
#               (pv * 100) / (config.RV_GAS_CONSTANT * Tk)
#         return rho
#
#     @staticmethod
#     def enrich_data(df):
#         print("Calculating Physics Metrics (Air Density & Power)...")
#
#         # 1. Calculate Air Density
#         df['Rho'] = df.apply(lambda row: RefineryPhysics.calculate_moist_density(
#             row['Temp'], row['RH']), axis=1)
#
#         # 2. Calculate Wind Power Density (WPD) = 0.5 * rho * v^3
#         df['WPD'] = 0.5 * df['Rho'] * (df['Speed'] ** 3)
#
#         # 3. Vectorize Direction (Convert 0-360 degrees to X/Y coordinates)
#         # This is essential so the AI knows that 359° is close to 1°
#         wd_rad = np.deg2rad(df['Direction'])
#         df['Wx'] = np.cos(wd_rad)
#         df['Wy'] = np.sin(wd_rad)
#
#         return df

# physics_engine.py
import numpy as np
import config


class RefineryPhysics:
    @staticmethod
    def calculate_moist_density(temp_c, rh):
        """
        Calculates density of moist air.
        Refineries are often hot, reducing air density and turbine efficiency.
        """
        # Convert Temp to Kelvin
        Tk = temp_c + 273.15

        # Saturation Vapor Pressure (Tetens Formula)
        es = 6.1078 * 10 ** ((7.5 * temp_c) / (temp_c + 237.3))

        # Vapor Pressure
        pv = es * (rh / 100.0)

        # Dry Air Pressure
        pdry = config.STD_PRESSURE - pv

        # Density Calculation (kg/m^3)
        rho = (pdry * 100) / (config.RD_GAS_CONSTANT * Tk) + \
              (pv * 100) / (config.RV_GAS_CONSTANT * Tk)
        return rho

    @staticmethod
    def extrapolate_wind_shear(df, alpha=0.35):
        """
        Predicts wind speeds at higher altitudes using Hellmann Power Law.
        Refineries have high 'Roughness Length' (tanks/structures), so alpha is high (~0.35).
        """
        print("   -> Extrapolating Vertical Wind Profile (10m -> 30m -> 50m)...")

        # Sensor usually at 10m. We predict for 30m (Small Turbine) and 50m (Mid-size)
        h_sensor = 10
        h_small = 30
        h_mid = 50

        # Save original as 10m
        df['Speed_10m'] = df['Speed']

        # Calculate speed at new heights: v_h = v_ref * (h / h_ref)^alpha
        df['Speed_30m'] = df['Speed'] * (h_small / h_sensor) ** alpha
        df['Speed_50m'] = df['Speed'] * (h_mid / h_sensor) ** alpha

        # Calculate Potential Power Density at new heights
        df['WPD_30m'] = 0.5 * df['Rho'] * (df['Speed_30m'] ** 3)
        df['WPD_50m'] = 0.5 * df['Rho'] * (df['Speed_50m'] ** 3)

        return df

    @staticmethod
    def enrich_data(df):
        print("Calculating Physics Metrics (Air Density, Power, & Shear)...")

        # 1. Calculate Air Density (The Old Logic - Kept)
        df['Rho'] = df.apply(lambda row: RefineryPhysics.calculate_moist_density(
            row['Temp'], row['RH']), axis=1)

        # 2. Calculate Wind Power Density (WPD) = 0.5 * rho * v^3
        df['WPD'] = 0.5 * df['Rho'] * (df['Speed'] ** 3)

        # 3. Vectorize Direction (The Old Logic - Kept)
        wd_rad = np.deg2rad(df['Direction'])
        df['Wx'] = np.cos(wd_rad)
        df['Wy'] = np.sin(wd_rad)

        # 4. NEW: Vertical Extrapolation (The "Go Higher" Strategy)
        df = RefineryPhysics.extrapolate_wind_shear(df)

        return df