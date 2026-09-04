import logging

import ismrmrd
import numpy as np
import pytest

from utils.utils import check_OR_arguments, MRD5Dto3D, display_diagnostic, normalise, send_original_images


# ---------------------------------------------------------------------------
# check_OR_arguments()
# ---------------------------------------------------------------------------
class TestCheckORArguments:

    def test_none_config_return_default(self):
        assert check_OR_arguments(None, arg_name='test', arg_type=str, arg_default='default') == 'default'

    def test_non_dict_config_return_default(self):
        assert check_OR_arguments("not a dict", arg_name='test', arg_type=str, arg_default='default') == 'default'

    def test_missing_parameters_key_returns_default(self):
        assert check_OR_arguments({"Arg": "value"}, arg_name='test', arg_type=str, arg_default='default') == 'default'

    def test_missing_arg_name_return_default(self):
        config = {"parameters":{"Arg": "value"}}
        assert check_OR_arguments(config, arg_name='test', arg_type=str, arg_default='default') == 'default'

    def test_default_is_none_when_not_specified(self):
        assert check_OR_arguments({}, arg_name='test', arg_type=str) == None

    # arg_type = str
    def test_str_type_value_passed_through(self):
        config = {"parameters": {"Arg": "value"}}
        assert check_OR_arguments(config, arg_name='Arg', arg_type=str, arg_default='default') == 'value'

    # arg_type = bool
    def test_bool_type_already_bool_passed_through(self):
        config = {"parameters": {"Arg": True}}
        assert check_OR_arguments(config, arg_name='Arg', arg_type=bool, arg_default=False) is True

    def test_bool_type_str_case_insensitive(self):
        for value in ("true", "True", "TRUE", "tRUe"):
            config = {"parameters": {"Arg": value}}
            assert check_OR_arguments(config, arg_name='Arg', arg_type=bool, arg_default=False) is True
        
        for value in ("false", "False", "FALSE", "fALse"):
            config = {"parameters": {"Arg": value}}
            assert check_OR_arguments(config, arg_name='Arg', arg_type=bool, arg_default=True) is False

    def test_bool_type_invalid_str_raises_value_error(self):
        config = {"parameters": {"Arg": "error"}}
        with pytest.raises(ValueError):
            check_OR_arguments(config, arg_name='Arg', arg_type=bool, arg_default=False)

    # arg_type = int
    def test_int_type_already_int_passed_through(self):
        config = {"parameters": {"Arg": 42}}
        assert check_OR_arguments(config, arg_name='Arg', arg_type=int, arg_default=0) == 42

    def test_int_type_str_cast_to_int(self):
        config = {"parameters": {"Arg": "42"}}
        assert check_OR_arguments(config, arg_name='Arg', arg_type=int, arg_default=0) == 42

    def test_int_type_invalid_str_raises_value_error(self):
        config = {"parameters": {"Arg": "error"}}
        with pytest.raises(ValueError):
            check_OR_arguments(config, arg_name='Arg', arg_type=int, arg_default=0)

    # arg_type = float
    def test_float_type_already_float_passed_through(self):
        config = {"parameters": {"Arg": 42.42}}
        assert check_OR_arguments(config, arg_name='Arg', arg_type=float, arg_default=0.0) == 42.42

    def test_float_type_str_cast_to_float(self):
        config = {"parameters": {"Arg": "42.42"}}
        assert check_OR_arguments(config, arg_name='Arg', arg_type=float, arg_default=0.0) == 42.42

    def test_float_type_invalid_str_raises_value_error(self):
        config = {"parameters": {"Arg": "error"}}
        with pytest.raises(ValueError):
            check_OR_arguments(config, arg_name='Arg', arg_type=int, arg_default=0.0)

    # unsupported type
    def test_unsupported_type_raises_type_error(self):
        config = {"parameters": {"Arg": [1, 2, 3]}}
        with pytest.raises(TypeError):
            check_OR_arguments(config, arg_name='Arg', arg_type=list, arg_default=None)


# ---------------------------------------------------------------------------
# send_original_images()
# ---------------------------------------------------------------------------
class TestSendOriginalImages:

    def test_forwards_all_images_to_connection_in_order(self, fake_connection, make_image):
        images = [make_image(slice=0), make_image(slice=1), make_image(slice=2)]
 
        send_original_images(images, fake_connection)
 
        assert fake_connection.sent_images == images
 
    def test_empty_list_sends_nothing(self, fake_connection):
        send_original_images([], fake_connection)
 
        assert fake_connection.sent_images == []

    def test_sets_keep_image_geometry_flag_in_meta(self, fake_connection, make_image):
        image = make_image()

        send_original_images([image], fake_connection)

        sent_meta = ismrmrd.Meta.deserialize(fake_connection.sent_images[0].attribute_string)
        assert sent_meta["Keep_image_geometry"] == "1"

    def test_preserves_existing_meta_fields(self, fake_connection, make_image):
        image = make_image()
        existing_meta = ismrmrd.Meta.deserialize(image.attribute_string)
        existing_meta["SequenceDescription"] = "Test"
        image.attribute_string = existing_meta.serialize()

        send_original_images([image], fake_connection)

        sent_meta = ismrmrd.Meta.deserialize(fake_connection.sent_images[0].attribute_string)
        assert sent_meta["SequenceDescription"] == "Test"
        assert sent_meta["Keep_image_geometry"] == "1"


# ---------------------------------------------------------------------------
# display_diagnostic()
# ---------------------------------------------------------------------------
class TestDisplayDiagnostic:

    def test_returns_expected_keys(self, make_header):
        head = make_header()
        meta = ismrmrd.Meta()

        result = display_diagnostic(head, meta)

        assert set(result.keys()) == {"matrix", "fov", "voxelsize", "read_dir", "phase_dir", "slice_dir"}

    def test_matrix_and_fov_match_header(self, make_header):
        head = make_header(matrix=(128, 64, 32), fov=(256.0, 128.0, 64.0))
        meta = ismrmrd.Meta()

        result = display_diagnostic(head, meta)

        assert list(result["matrix"]) == [128, 64, 32]
        assert list(result["fov"]) == [256.0, 128.0, 64.0]

    def test_does_not_crash_with_ice_mini_head_present(self, make_header):
        head = make_header()
        meta = ismrmrd.Meta()
        meta['IceMiniHead'] = "dGVzdA=="  # "test" in base64

        result = display_diagnostic(head, meta)

        assert result["matrix"] is not None

    def test_ice_mini_head_decoded_and_logged_when_flag_true(self, make_header, caplog):
        head = make_header()
        meta = ismrmrd.Meta()
        meta['IceMiniHead'] = "aGVsbG8="  # "hello" in base64

        with caplog.at_level(logging.INFO):
            display_diagnostic(head, meta, ICEminihead_decode=True)

        assert any("hello" in record.message for record in caplog.records)

    def test_ice_mini_head_not_decoded_when_flag_false(self, make_header, caplog):
        head = make_header()
        meta = ismrmrd.Meta()
        meta['IceMiniHead'] = "aGVsbG8="  # base64 for "hello"

        with caplog.at_level(logging.INFO):
            display_diagnostic(head, meta, ICEminihead_decode=False)

        assert not any("hello" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# normalise()
# ---------------------------------------------------------------------------
class TestNormalise:

    def test_scales_to_12_bit_max(self):
        data = np.array([0.0, 50.0, 100.0], dtype=np.float32)
        result = normalise(data)
        assert result.max() == 4095

    def test_max_value_maps_exactly_to_4095_and_zero_stays_zero(self):
        data = np.array([0.0, 10.0, 20.0, 55.0], dtype=np.float32)
        result = normalise(data)
        assert result[0] == 0
        assert result[-1] == 4095

    def test_relative_order_is_preserved(self):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = normalise(data)
        assert result[0] < result[1] < result[2] < result[3]

    def test_all_zero_input_raises_zero_division_error(self):
        data = np.zeros(3, dtype=np.float32)
        with pytest.raises(ZeroDivisionError):
            normalise(data)

    def test_integer_dtype_input_raises_type_error(self):
        data = np.array([0, 50, 100], dtype=np.int16)
        with pytest.raises(TypeError):
            normalise(data)

# ---------------------------------------------------------------------------
# MRD5Dto3D()
# ---------------------------------------------------------------------------
class TestMRD5Dto3D:

    def test_output_shape(self):
        # [img, cha, z, y, x] = [3, 1, 1, 4, 5]
        data = np.zeros((3, 1, 1, 4, 5), dtype=np.float32)
        result = MRD5Dto3D(data)
        assert result.shape == (4, 5, 3)  # [y, x, img]

    def test_keeps_first_channel_and_first_slice_values(self):
        n_img = 3
        data = np.zeros((n_img, 2, 2, 4, 5), dtype=np.float32)
        for i in range(n_img):
            data[i, 0, 0, :, :] = i + 1
            data[i, 1, 0, :, :] = 999
            data[i, 0, 1, :, :] = 888

        result = MRD5Dto3D(data)

        for i in range(n_img):
            assert np.all(result[:, :, i] == i + 1)

    def test_wrong_ndim_raises_value_error(self):
        data = np.zeros((3, 4, 5), dtype=np.float32)  # missing the [img, cha] axes
        with pytest.raises(ValueError):
            MRD5Dto3D(data)

