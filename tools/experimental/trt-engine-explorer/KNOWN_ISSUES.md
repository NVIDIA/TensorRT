# Known Limitations and Issues

* TensorRT 9.0 uses `kgen` kernels which produce broken representations of JSON engine graphs and no per-layer timing or inference profiling information.

* TREx does not always display the correct region shapes. This is due to missing information in the graph JSON produced by TensorRT. This will be fixed in a future TensorRT release.
* TREx has partial support for multi-profile engines. You can choose which profile to use for the engine plan, but profiling information will only load when using the first (default) profile (0). For example, to load profile 3:

    ```
    plan = EnginePlan(
        graph_file="my_engine_name.graph.json",
        profiling_file="my_engine.profile.json",
        profile_id=3)

    ```
    TRex will emit a warning saying that the profiling data was dropped.

    To use the profiling data, load profile 0:

    ```
    plan = EnginePlan(
        graph_file="my_engine_name.graph.json",
        profiling_file="my_engine.profile.json",
        profile_id=0)

    ```
* `display_df` may hang or return with an error (perhaps after a long delay).
<br><br>
This is a known `dtale` issue which may be caused by firewalls, VPNs or other network connectivity issues, but it may also be due to a problem when reading the server's hostname.<br>
You can try to explicitly configure the hostname in `dtale`. For example, if you are running the Jupyter server on a machine that has IP address 192.168.1.30:
    ```
    dtale.app.ACTIVE_HOST = "192.168.1.30"
    ```
    **Note**: you should configure the host address before trying to use any of the functions that display tables. For example: configure `dtale.app.ACTIVE_HOST` right after importing `trex` or `dtale` at the top of your notebook.
    <br>
    Alternatively, you can change the backend library used for rendering tables by setting `trex.set_table_display_backend` to one of the other valid backends.
    <br>
    To use the `qgrid` backend:
    ```
    trex.set_table_display_backend(display_df_qgrid)
    ```

    To use the `IPython` backend:
    ```
    trex.set_table_display_backend(display)
    ```
    **Further details:** When `display_df` is invoked, TREx forwards the function call to one of it table-rendering backends. The default backend is `dtale`, which uses a Flask server to render Pandas dataframes as sophisticated HTML tables. `dtale` attempts to detect the address of the flask server by using `socket.gethostname()`, which may fail in some systems. Upon such failure, explicitly setting `dtale.app.ACTIVE_HOST` to the Flask server's IP address (or name) should help.
