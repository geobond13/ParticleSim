"""
Tests for Intake Geometry Module

Validates:
- Clausing transmission factor
- Angle-dependent transmission
- Honeycomb intake geometry
- Freestream velocity sampling
- Attitude jitter

Week 4 Deliverable - IntakeSIM
"""

import pytest
import numpy as np
from intakesim.geometry import (
    clausing_factor_analytical,
    transmission_probability_angle,
    HoneycombIntake,
    sample_freestream_velocity,
    apply_attitude_jitter,
    compute_compression_ratio,
    compute_hexagonal_channel_centers,
    get_channel_id_from_position,
    get_radial_distance_from_channel_center,
    get_wall_normal_at_position,
)


class TestClausingFactor:
    """Test Clausing transmission factor calculations."""

    def test_zero_length_tube(self):
        """L/D = 0 should give perfect transmission (K=1)."""
        K = clausing_factor_analytical(0.0)
        assert abs(K - 1.0) < 1e-10, "Zero-length tube should have K=1"

    def test_clausing_decreases_with_length(self):
        """Transmission should decrease as tube gets longer."""
        K_short = clausing_factor_analytical(5.0)
        K_long = clausing_factor_analytical(50.0)

        assert K_short > K_long, "Longer tubes should have lower transmission"

    def test_clausing_asymptotic_limit(self):
        """For L/D >> 1, K ≈ 8D/(3L) = 8/(3×L/D)."""
        L_over_D = 100.0
        K = clausing_factor_analytical(L_over_D)
        K_asymptotic = 8.0 / (3.0 * L_over_D)

        print(f"\n  K(L/D=100) = {K:.4f}, Asymptotic = {K_asymptotic:.4f}")

        # Should be close to asymptotic limit
        assert abs(K - K_asymptotic) / K_asymptotic < 0.1, \
            "Long tube should match asymptotic formula"

    def test_clausing_typical_values(self):
        """Test Clausing factors for typical L/D ratios."""
        test_cases = [
            (10.0, 0.03, 0.08),   # L/D=10, K ~ 0.047 (empirical fit)
            (20.0, 0.01, 0.02),   # L/D=20, K ~ 0.013 (empirical fit)
            (50.0, 0.001, 0.005), # L/D=50, K ~ 0.002 (empirical fit)
        ]

        for L_over_D, K_min, K_max in test_cases:
            K = clausing_factor_analytical(L_over_D)
            print(f"  K(L/D={L_over_D}) = {K:.4f}")
            assert K_min < K < K_max, f"K for L/D={L_over_D} outside expected range"


class TestAngleDependentTransmission:
    """Test angle-dependent transmission probability."""

    def test_normal_incidence_maximum(self):
        """Normal incidence (θ=0) should give maximum transmission."""
        L_over_D = 20.0
        P_normal = transmission_probability_angle(0.0, L_over_D)
        P_oblique = transmission_probability_angle(np.pi/6, L_over_D)  # 30°

        assert P_normal > P_oblique, \
            "Normal incidence should have higher transmission than oblique"

    def test_grazing_incidence_zero(self):
        """Grazing incidence should give zero transmission."""
        L_over_D = 20.0
        theta_grazing = np.pi / 2  # 90°

        P_grazing = transmission_probability_angle(theta_grazing, L_over_D)

        assert P_grazing < 0.01, "Grazing incidence should have near-zero transmission"

    def test_max_acceptance_angle(self):
        """Particles beyond θ_max = atan(D/L) should not transmit."""
        L_over_D = 10.0
        theta_max = np.atan(1.0 / L_over_D)

        P_inside = transmission_probability_angle(theta_max * 0.9, L_over_D)
        P_outside = transmission_probability_angle(theta_max * 1.1, L_over_D)

        assert P_inside > 0, "Inside acceptance cone should transmit"
        assert P_outside == 0, "Outside acceptance cone should not transmit"


class TestHoneycombIntake:
    """Test honeycomb intake geometry."""

    def test_intake_initialization(self):
        """Honeycomb intake should initialize correctly."""
        inlet_area = 0.01  # 0.01 m² = 100 cm²
        outlet_area = 0.001  # 0.001 m² = 10 cm²
        channel_length = 0.1  # 10 cm
        channel_diameter = 0.001  # 1 mm

        intake = HoneycombIntake(inlet_area, outlet_area, channel_length, channel_diameter)

        print(f"\n  {intake}")
        print(f"  Number of channels: {intake.n_channels}")
        print(f"  Clausing factor: {intake.clausing_factor:.3f}")

        assert intake.n_channels > 0, "Should have positive number of channels"
        assert 0 < intake.clausing_factor < 1, "Clausing factor should be in (0,1)"

    def test_geometric_compression(self):
        """Geometric compression ratio should match area ratio."""
        inlet_area = 0.01
        outlet_area = 0.001
        channel_length = 0.1
        channel_diameter = 0.001

        intake = HoneycombIntake(inlet_area, outlet_area, channel_length, channel_diameter)

        expected_CR_geo = inlet_area / outlet_area

        assert abs(intake.geometric_compression - expected_CR_geo) < 1e-10, \
            "Geometric compression should equal area ratio"


class TestFreestreamInjection:
    """Test freestream velocity sampling."""

    def test_freestream_velocity_magnitude(self):
        """Freestream velocities should have correct mean magnitude."""
        v_orbital = 7780.0  # m/s (225 km altitude)
        T_atm = 900.0  # K
        mass = 16.0 * 1.66e-27  # Atomic oxygen mass
        n_samples = 1000

        v = sample_freestream_velocity(v_orbital, T_atm, mass, n_samples)

        # Mean velocity should be approximately -v_orbital in z-direction
        v_z_mean = np.mean(v[:, 2])

        print(f"\n  Mean v_z: {v_z_mean:.1f} m/s, Expected: {-v_orbital:.1f} m/s")

        # Within 10% due to thermal component
        assert abs(v_z_mean + v_orbital) / v_orbital < 0.1, \
            "Mean axial velocity should be approximately -v_orbital"

    def test_freestream_thermal_component(self):
        """Freestream should have thermal component."""
        v_orbital = 7780.0
        T_atm = 900.0
        mass = 16.0 * 1.66e-27
        n_samples = 5000

        v = sample_freestream_velocity(v_orbital, T_atm, mass, n_samples)

        # Standard deviation in perpendicular directions should be ~v_thermal
        std_x = np.std(v[:, 0])
        std_y = np.std(v[:, 1])

        # Expected thermal velocity
        v_thermal = np.sqrt(2.0 * 1.38e-23 * T_atm / mass)

        print(f"\n  Thermal velocity (expected): {v_thermal:.1f} m/s")
        print(f"  std(v_x): {std_x:.1f} m/s")
        print(f"  std(v_y): {std_y:.1f} m/s")

        # Should be of similar order
        assert abs(std_x - v_thermal) / v_thermal < 0.3, \
            "Perpendicular thermal spread should match expected"


class TestAttitudeJitter:
    """Test attitude jitter application."""

    def test_jitter_perturbs_velocity(self):
        """Jitter should perturb velocity vectors."""
        n_particles = 100
        v = np.zeros((n_particles, 3), dtype=np.float64)
        v[:, 2] = -7780.0  # All in -z direction

        jitter_angle_deg = 7.0

        v_jittered = apply_attitude_jitter(v, jitter_angle_deg)

        # After jitter, should have non-zero x and y components
        assert np.any(v_jittered[:, 0] != 0), "Jitter should affect x-component"
        assert np.any(v_jittered[:, 1] != 0), "Jitter should affect y-component"

    def test_zero_jitter_no_change(self):
        """Zero jitter should not change velocities."""
        n_particles = 10
        v = np.random.randn(n_particles, 3) * 1000

        v_jittered = apply_attitude_jitter(v, jitter_angle_deg=0.0)

        np.testing.assert_array_almost_equal(v, v_jittered, decimal=10,
                                              err_msg="Zero jitter should not change velocities")


class TestCompressionRatio:
    """Test compression ratio calculations."""

    def test_compression_ratio_basic(self):
        """Compression ratio should be n_outlet / n_inlet."""
        n_inlet = 1e20  # m^-3
        n_outlet = 5e21  # m^-3
        v_inlet = 7780.0  # m/s
        v_outlet = 1556.0  # m/s

        CR = compute_compression_ratio(n_inlet, n_outlet, v_inlet, v_outlet)

        expected_CR = n_outlet / n_inlet

        assert abs(CR - expected_CR) < 1e-10, "CR should equal n_outlet / n_inlet"


class TestMultiChannelGeometry:
    """
    Test multi-channel honeycomb geometry (Phase II).

    Validates:
    - Hexagonal packing of channel centers
    - Channel ID lookup from position
    - Radial distance calculations
    - Wall normal vector computations
    """

    def test_channel_centers_hexagonal_packing(self):
        """Channel centers should form proper hexagonal lattice."""
        inlet_area = 0.01  # 0.01 m² = 100 cm²
        channel_diameter = 0.001  # 1 mm
        n_channels_approx = int(inlet_area / (np.pi * (channel_diameter/2)**2))

        channel_centers = compute_hexagonal_channel_centers(
            n_channels_approx, channel_diameter, inlet_area
        )

        print(f"\n  Expected channels: {n_channels_approx}")
        print(f"  Generated channels: {len(channel_centers)}")

        # Should generate approximately the expected number of channels (within 20%)
        assert 0.8 * n_channels_approx <= len(channel_centers) <= 1.2 * n_channels_approx, \
            "Hexagonal packing should generate approximately expected number of channels"

        # All centers should be within inlet radius
        R_inlet = np.sqrt(inlet_area / np.pi)
        for i, center in enumerate(channel_centers[:10]):  # Check first 10
            r = np.sqrt(center[0]**2 + center[1]**2)
            assert r < R_inlet, f"Channel {i} center outside inlet radius"

        # Nearest neighbor spacing should be approximately channel_diameter
        if len(channel_centers) > 1:
            center_0 = channel_centers[0]
            distances = [np.sqrt((c[0]-center_0[0])**2 + (c[1]-center_0[1])**2)
                         for c in channel_centers[1:]]
            min_dist = min(distances)
            print(f"  Minimum neighbor distance: {min_dist*1000:.3f} mm")
            print(f"  Expected (channel_diameter): {channel_diameter*1000:.3f} mm")

            # Nearest neighbor should be ~1 channel diameter away (hexagonal packing)
            assert 0.8 * channel_diameter <= min_dist <= 1.2 * channel_diameter, \
                "Nearest neighbor spacing should match channel diameter"

    def test_get_channel_id_inside_channel(self):
        """Particle inside channel should return valid channel ID."""
        inlet_area = 0.01
        channel_diameter = 0.001
        channel_radius = channel_diameter / 2.0
        n_channels = int(inlet_area / (np.pi * channel_radius**2))

        channel_centers = compute_hexagonal_channel_centers(
            n_channels, channel_diameter, inlet_area
        )

        # Test point at center of first channel
        y0, z0 = channel_centers[0]
        channel_id = get_channel_id_from_position(y0, z0, channel_centers, channel_radius)

        print(f"\n  Position: ({y0:.4f}, {z0:.4f})")
        print(f"  Channel ID: {channel_id}")

        assert channel_id == 0, "Point at channel center should return correct channel ID"

        # Test point slightly offset from center (but still inside)
        y_offset = y0 + 0.3 * channel_radius
        z_offset = z0 + 0.3 * channel_radius
        channel_id_offset = get_channel_id_from_position(
            y_offset, z_offset, channel_centers, channel_radius
        )

        assert channel_id_offset == 0, "Point inside channel should return channel ID 0"

    def test_get_channel_id_outside_all_channels(self):
        """Particle outside all channels should return -1."""
        inlet_area = 0.01
        channel_diameter = 0.001
        channel_radius = channel_diameter / 2.0
        n_channels = int(inlet_area / (np.pi * channel_radius**2))

        channel_centers = compute_hexagonal_channel_centers(
            n_channels, channel_diameter, inlet_area
        )

        # Point far outside inlet area
        y_outside = 10.0 * np.sqrt(inlet_area)  # Way outside
        z_outside = 10.0 * np.sqrt(inlet_area)

        channel_id = get_channel_id_from_position(
            y_outside, z_outside, channel_centers, channel_radius
        )

        print(f"\n  Position outside: ({y_outside:.2f}, {z_outside:.2f})")
        print(f"  Channel ID: {channel_id}")

        assert channel_id == -1, "Point outside all channels should return -1"

    def test_radial_distance_at_channel_center(self):
        """Radial distance at channel center should be zero."""
        channel_center_y = 0.005  # 5 mm
        channel_center_z = 0.003  # 3 mm

        # Point exactly at center
        r_perp = get_radial_distance_from_channel_center(
            channel_center_y, channel_center_z,
            channel_center_y, channel_center_z
        )

        print(f"\n  Radial distance at center: {r_perp:.10f} m")

        assert abs(r_perp) < 1e-15, "Radial distance at channel center should be zero"

    def test_radial_distance_at_wall(self):
        """Radial distance at wall should equal channel radius."""
        channel_center_y = 0.0
        channel_center_z = 0.0
        channel_radius = 0.0005  # 0.5 mm

        # Point at wall (radial direction)
        y_wall = channel_center_y + channel_radius
        z_wall = channel_center_z

        r_perp = get_radial_distance_from_channel_center(
            y_wall, z_wall,
            channel_center_y, channel_center_z
        )

        print(f"\n  Channel radius: {channel_radius*1000:.3f} mm")
        print(f"  Radial distance: {r_perp*1000:.3f} mm")

        assert abs(r_perp - channel_radius) < 1e-10, \
            "Radial distance at wall should equal channel radius"

        # Also test at wall in different direction (diagonal)
        y_wall_diag = channel_center_y + channel_radius / np.sqrt(2)
        z_wall_diag = channel_center_z + channel_radius / np.sqrt(2)

        r_perp_diag = get_radial_distance_from_channel_center(
            y_wall_diag, z_wall_diag,
            channel_center_y, channel_center_z
        )

        assert abs(r_perp_diag - channel_radius) < 1e-10, \
            "Radial distance at wall (diagonal) should equal channel radius"

    def test_wall_normal_points_radially_outward(self):
        """Wall normal should point radially away from channel center."""
        channel_center_y = 0.0
        channel_center_z = 0.0

        # Test point to the right (+y direction)
        y_right = channel_center_y + 0.0005
        z_right = channel_center_z

        normal_right = get_wall_normal_at_position(
            y_right, z_right,
            channel_center_y, channel_center_z
        )

        print(f"\n  Position: y={y_right*1000:.3f} mm, z={z_right*1000:.3f} mm")
        print(f"  Normal: [{normal_right[0]:.3f}, {normal_right[1]:.3f}, {normal_right[2]:.3f}]")

        # Normal should point in +y direction
        assert abs(normal_right[0]) < 1e-10, "Normal should have no x-component"
        assert abs(normal_right[1] - 1.0) < 1e-10, "Normal should point in +y"
        assert abs(normal_right[2]) < 1e-10, "Normal should have no z-component"

        # Test point diagonal (both y and z positive)
        y_diag = channel_center_y + 0.0003
        z_diag = channel_center_z + 0.0004

        normal_diag = get_wall_normal_at_position(
            y_diag, z_diag,
            channel_center_y, channel_center_z
        )

        # Normal should be unit vector
        norm_mag = np.sqrt(normal_diag[0]**2 + normal_diag[1]**2 + normal_diag[2]**2)
        assert abs(norm_mag - 1.0) < 1e-10, "Normal should be unit vector"

        # Normal should point radially outward (positive y and z components)
        assert normal_diag[1] > 0, "Normal y-component should be positive"
        assert normal_diag[2] > 0, "Normal z-component should be positive"

    def test_honeycomb_intake_multichannel_mode(self):
        """HoneycombIntake with use_multichannel=True should compute channel centers."""
        inlet_area = 0.01
        outlet_area = 0.001
        channel_length = 0.02  # 20 mm
        channel_diameter = 0.001  # 1 mm

        # Legacy mode (default)
        intake_legacy = HoneycombIntake(
            inlet_area, outlet_area, channel_length, channel_diameter,
            use_multichannel=False
        )

        assert intake_legacy.channel_centers is None, \
            "Legacy mode should not compute channel centers"
        assert "tapered-cone" in str(intake_legacy), \
            "Legacy mode should report tapered-cone"

        # Multi-channel mode
        intake_multi = HoneycombIntake(
            inlet_area, outlet_area, channel_length, channel_diameter,
            use_multichannel=True
        )

        assert intake_multi.channel_centers is not None, \
            "Multi-channel mode should compute channel centers"
        assert len(intake_multi.channel_centers) > 0, \
            "Should have non-zero channel centers"
        assert "multi-channel" in str(intake_multi), \
            "Multi-channel mode should report multi-channel"

        print(f"\n  Legacy: {intake_legacy}")
        print(f"  Multi-channel: {intake_multi}")
        print(f"  Channel centers generated: {len(intake_multi.channel_centers)}")

    def test_honeycomb_intake_get_channel_id_method(self):
        """HoneycombIntake.get_channel_id() should return correct channel."""
        inlet_area = 0.01
        outlet_area = 0.001
        channel_length = 0.02
        channel_diameter = 0.001

        intake = HoneycombIntake(
            inlet_area, outlet_area, channel_length, channel_diameter,
            use_multichannel=True
        )

        # Get first channel center
        y0, z0 = intake.channel_centers[0]

        # Query channel ID at this position
        channel_id = intake.get_channel_id(y0, z0)

        print(f"\n  First channel center: ({y0:.4f}, {z0:.4f})")
        print(f"  Returned channel ID: {channel_id}")

        assert channel_id == 0, "Should return channel ID 0 for first channel center"

        # Test error handling for legacy mode
        intake_legacy = HoneycombIntake(
            inlet_area, outlet_area, channel_length, channel_diameter,
            use_multichannel=False
        )

        with pytest.raises(ValueError, match="use_multichannel=True"):
            intake_legacy.get_channel_id(0.0, 0.0)

    def test_honeycomb_intake_get_radial_distance_method(self):
        """HoneycombIntake.get_radial_distance() should return correct distance."""
        inlet_area = 0.01
        outlet_area = 0.001
        channel_length = 0.02
        channel_diameter = 0.001

        intake = HoneycombIntake(
            inlet_area, outlet_area, channel_length, channel_diameter,
            use_multichannel=True
        )

        # Position at first channel center
        y0, z0 = intake.channel_centers[0]
        pos_center = np.array([0.01, y0, z0])  # x=10mm (arbitrary)

        r_perp = intake.get_radial_distance(pos_center, channel_id=0)

        print(f"\n  Position at channel 0 center")
        print(f"  Radial distance: {r_perp*1e6:.3f} um")

        assert abs(r_perp) < 1e-12, "Distance at channel center should be zero"

        # Position offset from center
        y_offset = y0 + 0.0003  # 0.3 mm offset
        pos_offset = np.array([0.01, y_offset, z0])

        r_perp_offset = intake.get_radial_distance(pos_offset, channel_id=0)

        print(f"  Offset position: y={y_offset*1000:.3f} mm")
        print(f"  Radial distance: {r_perp_offset*1000:.3f} mm")

        assert abs(r_perp_offset - 0.0003) < 1e-10, \
            "Radial distance should match offset"

    def test_honeycomb_intake_get_wall_normal_method(self):
        """HoneycombIntake.get_wall_normal() should return correct normal vector."""
        inlet_area = 0.01
        outlet_area = 0.001
        channel_length = 0.02
        channel_diameter = 0.001

        intake = HoneycombIntake(
            inlet_area, outlet_area, channel_length, channel_diameter,
            use_multichannel=True
        )

        # Position offset in +y direction from channel 0
        y0, z0 = intake.channel_centers[0]
        pos = np.array([0.01, y0 + 0.0004, z0])

        normal = intake.get_wall_normal(pos, channel_id=0)

        print(f"\n  Position: y={pos[1]*1000:.3f} mm, z={pos[2]*1000:.3f} mm")
        print(f"  Normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")

        # Normal should point in +y direction (radially outward)
        assert abs(normal[0]) < 1e-10, "No x-component"
        assert normal[1] > 0.99, "Should point mostly in +y"
        assert abs(normal[2]) < 1e-10, "No z-component (offset only in y)"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
