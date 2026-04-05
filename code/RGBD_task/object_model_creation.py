import pyzed.sl as sl
import sys


def main():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode        = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.coordinate_units  = sl.UNIT.METER

    filter_params = sl.MeshFilterParameters()
    filter_params.set(sl.MESH_FILTER.LOW)

    print("Opening ZED camera...")                         # ← startup message
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open:", repr(err), "- Exit program.")
        exit(1)
    print("Camera opened successfully.\n")

    py_transform = sl.Transform()
    tracking_parameters = sl.PositionalTrackingParameters(_init_pos=py_transform)
    err = zed.enable_positional_tracking(tracking_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Enable positional tracking:", repr(err), "- Exit program.")
        zed.close()
        exit(1)

    mapping_parameters = sl.SpatialMappingParameters(
        map_type     = sl.SPATIAL_MAP_TYPE.MESH,
        save_texture = True,
    )
    err = zed.enable_spatial_mapping(mapping_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Enable spatial mapping:", repr(err), "- Exit program.")
        zed.close()
        exit(1)

    print("Starting capture — move slowly around your object...\n")  # ← capture start

    i = 0
    mesh = sl.Mesh()
    runtime_parameters = sl.RuntimeParameters()

    while i < 500:
        if i % 50 == 0:
            print(f"{i} / 500")
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            mapping_state = zed.get_spatial_mapping_state()
            sys.stdout.write("Images captured: {0} / 500 || {1} \033[K\r".format(i, mapping_state))
            sys.stdout.flush()
            i += 1

    print("\n\nCapture complete.")

    print("Extracting mesh...")
    err = zed.extract_whole_spatial_map(mesh)
    print(repr(err))

    print("Filtering mesh...")
    mesh.filter(filter_params)

    print("Applying texture...")
    mesh.apply_texture()

    print("Saving mesh...")
    mesh.save("mesh.obj")
    print("Saved → mesh.obj")

    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()


if __name__ == "__main__":
    main()