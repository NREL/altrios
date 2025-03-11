use crate::imports::*;
mod altrios_api_utils;
use crate::utilities::parse_ts_as_fn_defs;
use altrios_api_utils::*;

pub(crate) fn altrios_api(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut ast = syn::parse_macro_input!(item as syn::ItemStruct);
    let ident = &ast.ident;
    // println!("{}", String::from("*").repeat(30));
    // println!("struct: {}", ast.ident.to_string());

    let mut py_impl_block = TokenStream2::default();

    py_impl_block.extend::<TokenStream2>(parse_ts_as_fn_defs(attr, vec![], false, vec![]));

    if let syn::Fields::Named(syn::FieldsNamed { named, .. }) = &mut ast.fields {
        process_named_field_structs(named, &mut py_impl_block);
    } else if let syn::Fields::Unnamed(syn::FieldsUnnamed { unnamed, .. }) = &mut ast.fields {
        process_tuple_struct(unnamed, &mut py_impl_block);
    } else {
        abort_call_site!(
            "Invalid use of `altrios_api` macro.  Expected tuple struct or C-style struct."
        );
    };

    py_impl_block.extend::<TokenStream2>(quote! {
        #[staticmethod]
        #[pyo3(name = "default")]
        /// Exposes `default` to python.
        fn default_py() -> anyhow::Result<Self> {
            Ok(Self::default())
        }

        /// See [SerdeAPI::to_str]
        #[pyo3(name = "to_str")]
        pub fn to_str_py(&self, format: &str) -> anyhow::Result<String> {
            self.to_str(format)
        }

        /// Read (deserialize) an object from a string
        ///
        /// # Arguments:
        ///
        /// * `contents`: `str` - The string containing the object data
        /// * `format`: `str` - The source format, any of those listed in [`ACCEPTED_STR_FORMATS`](`SerdeAPI::ACCEPTED_STR_FORMATS`)
        ///
        #[staticmethod]
        #[pyo3(name = "from_str")]
        #[pyo3(signature = (contents, format, skip_init=None))]
        pub fn from_str_py(contents: &str, format: &str, skip_init: Option<bool>) -> PyResult<Self> {
            Ok(SerdeAPI::from_str(contents, format, skip_init.unwrap_or_default())?)
        }

        /// See [SerdeAPI::to_json]
        #[pyo3(name = "to_json")]
        fn to_json_py(&self) -> anyhow::Result<String> {
            self.to_json()
        }

        /// Read (deserialize) an object from a JSON string
        ///
        /// # Arguments
        ///
        /// * `json_str`: `str` - JSON-formatted string to deserialize from
        ///
        #[staticmethod]
        #[pyo3(name = "from_json")]
        #[pyo3(signature = (json_str, skip_init=None))]
        pub fn from_json_py(json_str: &str, skip_init: Option<bool>) -> PyResult<Self> {
            Ok(Self::from_json(json_str, skip_init.unwrap_or_default())?)
        }

        /// See [SerdeAPI::to_yaml]
        #[pyo3(name = "to_yaml")]
        fn to_yaml_py(&self) -> anyhow::Result<String> {
            self.to_yaml()
        }

        /// Read (deserialize) an object from a YAML string
        ///
        /// # Arguments
        ///
        /// * `yaml_str`: `str` - YAML-formatted string to deserialize from
        ///
        #[staticmethod]
        #[pyo3(name = "from_yaml")]
        #[pyo3(signature = (yaml_str, skip_init=None))]
        pub fn from_yaml_py(yaml_str: &str, skip_init: Option<bool>) -> PyResult<Self> {
            Ok(Self::from_yaml(yaml_str, skip_init.unwrap_or_default())?)
        }

        /// Write (serialize) an object to a message pack
        #[cfg(feature = "msgpack")]
        #[pyo3(name = "to_msg_pack")]
        pub fn to_msg_pack_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
            Ok(PyBytes::new_bound(py, &self.to_msg_pack().map_err(|e| PyIOError::new_err(format!("{:?}", e)))?))
        }

        /// Read (deserialize) an object from a message pack
        ///
        /// # Arguments
        /// * `msg_pack`: message pack
        #[cfg(feature = "msgpack")]
        #[staticmethod]
        #[pyo3(name = "from_msg_pack")]
        #[pyo3(signature = (msg_pack, skip_init=None))]
        pub fn from_msg_pack_py(msg_pack: &Bound<PyBytes>, skip_init: Option<bool>) -> PyResult<Self> {
            Self::from_msg_pack(
                msg_pack.as_bytes(),
                skip_init.unwrap_or_default()
            ).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        #[pyo3(name = "init")]
        fn init_py(&mut self) -> PyResult<()> {
            self.init().map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// `__copy__` magic method that uses `clone`.
        fn __copy__(&self) -> Self {
            self.clone()
        }

        /// `__deepcopy__` magic method that uses `clone`.
        fn __deepcopy__(&self) -> Self {
            self.clone()
        }

        #[pyo3(name = "clone")]
        /// calls Rust's `clone`.
        fn clone_py(&self) -> Self {
            self.clone()
        }
    });

    let py_impl_block = quote! {
        #[pymethods]
        #[allow(non_snake_case)]
        #[cfg(feature="pyo3")]
        /// Implement methods exposed and used in Python via PyO3
        impl #ident {
            #py_impl_block

            /// Write (serialize) an object to a file.
            /// Supported file extensions are listed in [`ACCEPTED_BYTE_FORMATS`](`SerdeAPI::ACCEPTED_BYTE_FORMATS`).
            /// Creates a new file if it does not already exist, otherwise truncates the existing file.
            ///
            /// # Arguments
            ///
            /// * `filepath`: `str | pathlib.Path` - The filepath at which to write the object
            ///
            #[pyo3(name = "to_file")]
            pub fn to_file_py(&self, filepath: &Bound<PyAny>) -> anyhow::Result<()> {
                self.to_file(PathBuf::extract_bound(filepath)?)
            }

            /// Read (deserialize) an object from a file.
            /// Supported file extensions are listed in [`ACCEPTED_BYTE_FORMATS`](`SerdeAPI::ACCEPTED_BYTE_FORMATS`).
            ///
            /// # Arguments:
            ///
            /// * `filepath`: `str | pathlib.Path` - The filepath from which to read the object
            ///
            #[staticmethod]
            #[pyo3(name = "from_file")]
            #[pyo3(signature = (filepath, skip_init=None))]
            pub fn from_file_py(filepath: &Bound<PyAny>, skip_init: Option<bool>) -> PyResult<Self> {
                Self::from_file(
                    PathBuf::extract_bound(filepath)?,
                    skip_init.unwrap_or_default()
                ).map_err(|err| PyIOError::new_err(format!("{:?}", err)))
            }
        }
    };
    let mut final_output = TokenStream2::default();
    final_output.extend::<TokenStream2>(quote! {
        #[cfg_attr(feature="pyo3", pyclass(module="altrios_pyo3", subclass, eq))]
    });
    let mut output: TokenStream2 = ast.to_token_stream();
    output.extend(py_impl_block);
    // println!("{}", output.to_string());
    final_output.extend::<TokenStream2>(output);
    final_output.into()
}

fn process_named_field_structs(
    named: &mut syn::punctuated::Punctuated<syn::Field, syn::token::Comma>,
    py_impl_block: &mut TokenStream2,
) {
    // struct with named fields
    for field in named.iter_mut() {
        let ftype = field.ty.clone();

        // if attr.tokens.to_string().contains("skip_get"){
        // for (i, idx_del) in idxs_del.into_iter().enumerate() {
        //     attr_vec.remove(*idx_del - i);
        // }

        // this is my quick and dirty attempt at emulating:
        // https://github.com/PyO3/pyo3/blob/48690525e19b87818b59f99be83f1e0eb203c7d4/pyo3-macros-backend/src/pyclass.rs#L220

        let mut opts = FieldOptions::default();
        let keep: Vec<bool> = field
            .attrs
            .iter()
            .map(|attr| {
                if let Meta::List(ml) = &attr.meta {
                    // catch the `api` in `#[api(skip_get)]`
                    if ml.path.is_ident("api") {
                        let opt_str = ml.tokens.to_string();
                        let opt_split = opt_str.as_str().split(",");
                        let mut opt_vec = opt_split.map(|opt| opt.trim()).collect::<Vec<&str>>();

                        // find the `skip_get` option
                        let mut idx_skip_get: Option<usize> = None;
                        opt_vec.iter().enumerate().for_each(|(i, opt)| {
                            if *opt == "skip_get" {
                                idx_skip_get = Some(i);
                                opts.skip_get = true;
                            }
                        });
                        if let Some(idx_skip_get) = idx_skip_get {
                            let _ = opt_vec.remove(idx_skip_get);
                        }

                        // find the `skip_set` option
                        let mut idx_skip_set: Option<usize> = None;
                        opt_vec.iter().enumerate().for_each(|(i, opt)| {
                            if *opt == "skip_set" {
                                idx_skip_set = Some(i);
                                opts.skip_set = true;
                            }
                        });
                        if let Some(idx_skip_set) = idx_skip_set {
                            let _ = opt_vec.remove(idx_skip_set);
                        }

                        // make sure there were no invalid options passed and raise warning
                        if !opt_vec.is_empty() {
                            emit_error!(ml.span(), "Invalid option(s): {:?}", opt_vec);
                        }
                        false // this attribute should not be retained because it is handled solely by this proc macro
                    } else {
                        true
                    }
                } else {
                    true
                }
            })
            .collect();
        // println!("options {:?}", opts);
        // this drops attrs with `#[altrios_api(...)]`, removing the field attribute from the struct def
        // iter.for_each(|x| field.attrs.retain(|_| *x));
        let new_attrs: (Vec<&syn::Attribute>, Vec<bool>) = field
            .attrs
            .iter()
            .zip(keep.iter())
            .filter(|(_a, k)| **k)
            .unzip();
        field.attrs = new_attrs.0.iter().cloned().cloned().collect();

        impl_getters_and_setters(py_impl_block, field, &opts, &ftype);
    }
}

fn process_tuple_struct(
    unnamed: &mut syn::punctuated::Punctuated<syn::Field, syn::token::Comma>,
    py_impl_block: &mut TokenStream2,
) {
    // tuple struct
    assert!(unnamed.len() <= 2);
    let idx_for_vec = syn::Index::from(if unnamed.len() == 1 { 0 } else { 1 });
    let re = Regex::new(r"Vec < (.+) >").unwrap();
    for field in unnamed.iter() {
        let ftype = field.ty.clone();
        if let syn::Type::Path(type_path) = ftype.clone() {
            let type_str = type_path.clone().into_token_stream().to_string();
            if type_str.contains("Vec") {
                // println!("{}", type_str);
                // println!("{}", &re.captures(&type_str).unwrap()[1]);
                let contained_dtype: TokenStream2 = re.captures(&type_str).unwrap()[1]
                    .to_string()
                    .parse()
                    .unwrap();
                py_impl_block.extend::<TokenStream2>(
                    quote! {
                        /// Rust-defined `__repr__` magic method for Python used exposed via PyO3.
                        fn __repr__(&self) -> String {
                            format!("Pyo3Vec({:?})", self.#idx_for_vec)
                        }
                        /// Rust-defined `__str__` magic method for Python used exposed via PyO3.
                        fn __str__(&self) -> String {
                            format!("{:?}", self.#idx_for_vec)
                        }
                        /// Rust-defined `__getitem__` magic method for Python used exposed via PyO3.
                        /// Prevents the Python user getting item directly using indexing.
                        fn __getitem__(&self, _idx: usize) -> anyhow::Result<()> {
                            bail!(PyNotImplementedError::new_err(
                                "Getting Rust vector value at index is not implemented.
                            Run `tolist` method to convert to standalone Python list.",
                            ))
                        }
                        /// Rust-defined `__setitem__` magic method for Python used exposed via PyO3.
                        /// Prevents the Python user setting item using indexing.
                        fn __setitem__(&mut self, _idx: usize, _new_value: #contained_dtype) -> anyhow::Result<()> {
                            bail!(PyNotImplementedError::new_err(
                                "Setting list value at index is not implemented.
                            Run `tolist` method, modify value at index, and
                            then set entire list.",
                            ))
                        }
                        /// PyO3-exposed method to convert vec-containing struct to Python list.
                        fn tolist(&self) -> anyhow::Result<Vec<#contained_dtype>> {
                            Ok(self.#idx_for_vec.clone())
                        }
                        /// Rust-defined `__len__` magic method for Python used exposed via PyO3.
                        /// Returns the length of the Rust vector.
                        fn __len__(&self) -> usize {
                            self.#idx_for_vec.len()
                        }
                        /// PyO3-exposed method to check if the vec-containing struct is empty.
                        #[pyo3(name = "is_empty")]
                        fn is_empty_py(&self) -> bool {
                            self.#idx_for_vec.is_empty()
                        }
                    }
                );
            }
        }
    }
}

pub(crate) fn altrios_enum_api(attr: TokenStream, item: TokenStream) -> TokenStream {
    let py_impl_block = TokenStream2::default();
    let impl_block = TokenStream2::default();

    let ast = syn::parse_macro_input!(item as syn::ItemEnum);
    let ident = &ast.ident;
    let output: TokenStream2 = ast.to_token_stream();
    // println!("{}", String::from("*").repeat(30));
    // println!("struct: {}", ast.ident.to_string());

    process_pyclass_generic(py_impl_block, attr, ident, output, impl_block, false)
}

fn process_pyclass_generic(
    mut py_impl_block: TokenStream2,
    attr: TokenStream,
    ident: &Ident,
    mut output: TokenStream2,
    impl_block: TokenStream2,
    subclass: bool,
) -> TokenStream {
    py_impl_block.extend::<TokenStream2>(parse_ts_as_fn_defs(attr, vec![], false, vec![]));

    add_serde_methods(&mut py_impl_block);

    let py_impl_block = quote! {
        #[allow(non_snake_case)]
        #[pymethods]
        #[cfg(feature="pyo3")]
        /// Implement methods exposed and used in Python via PyO3
        impl #ident {
            #py_impl_block
        }
    };
    let mut final_output = TokenStream2::default();
    if subclass {
        final_output.extend::<TokenStream2>(quote! {
            #[cfg_attr(feature="pyo3", pyclass(module = "altrios", subclass, eq))]
        });
    } else {
        final_output.extend::<TokenStream2>(quote! {
            #[cfg_attr(feature="pyo3", pyclass(module = "altrios", eq))]
        });
    }
    output.extend(impl_block);
    output.extend(py_impl_block);
    // println!("{}", output.to_string());
    final_output.extend::<TokenStream2>(output);
    final_output.into()
}

fn add_serde_methods(py_impl_block: &mut TokenStream2) {
    // NOTE: may be helpful to add an `init_py` method
    py_impl_block.extend::<TokenStream2>(quote! {
        pub fn copy(&self) -> Self {self.clone()}
        pub fn __copy__(&self) -> Self {self.clone()}
        pub fn __deepcopy__(&self, _memo: &Bound<PyDict>) -> Self {self.clone()}

        /// Read (deserialize) an object from a resource file packaged with the `altrios-core` crate
        ///
        /// # Arguments:
        ///
        /// * `filepath`: `str | pathlib.Path` - Filepath, relative to the top of the `resources` folder (excluding any relevant prefix), from which to read the object
        ///
        #[cfg(feature = "resources")]
        #[staticmethod]
        #[pyo3(name = "from_resource")]
        #[pyo3(signature = (filepath, skip_init=None))]
        pub fn from_resource_py(filepath: &Bound<PyAny>, skip_init: Option<bool>) -> PyResult<Self> {
            Self::from_resource(PathBuf::extract_bound(filepath)?, skip_init.unwrap_or_default()).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Write (serialize) an object to a file.
        /// Supported file extensions are listed in [`ACCEPTED_BYTE_FORMATS`](`SerdeAPI::ACCEPTED_BYTE_FORMATS`).
        /// Creates a new file if it does not already exist, otherwise truncates the existing file.
        ///
        /// # Arguments
        ///
        /// * `filepath`: `str | pathlib.Path` - The filepath at which to write the object
        ///
        #[pyo3(name = "to_file")]
        pub fn to_file_py(&self, filepath: &Bound<PyAny>) -> PyResult<()> {
           self.to_file(PathBuf::extract_bound(filepath)?).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Read (deserialize) an object from a file.
        /// Supported file extensions are listed in [`ACCEPTED_BYTE_FORMATS`](`SerdeAPI::ACCEPTED_BYTE_FORMATS`).
        ///
        /// # Arguments:
        ///
        /// * `filepath`: `str | pathlib.Path` - The filepath from which to read the object
        ///
        #[staticmethod]
        #[pyo3(name = "from_file")]
        #[pyo3(signature = (filepath, skip_init=None))]
        pub fn from_file_py(filepath: &Bound<PyAny>, skip_init: Option<bool>) -> PyResult<Self> {
            Self::from_file(PathBuf::extract_bound(filepath)?, skip_init.unwrap_or_default()).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Write (serialize) an object into a string
        ///
        /// # Arguments:
        ///
        /// * `format`: `str` - The target format, any of those listed in [`ACCEPTED_STR_FORMATS`](`SerdeAPI::ACCEPTED_STR_FORMATS`)
        ///
        #[pyo3(name = "to_str")]
        pub fn to_str_py(&self, format: &str) -> PyResult<String> {
            self.to_str(format).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Read (deserialize) an object from a string
        ///
        /// # Arguments:
        ///
        /// * `contents`: `str` - The string containing the object data
        /// * `format`: `str` - The source format, any of those listed in [`ACCEPTED_STR_FORMATS`](`SerdeAPI::ACCEPTED_STR_FORMATS`)
        ///
        #[staticmethod]
        #[pyo3(name = "from_str")]
        #[pyo3(signature = (contents, format, skip_init=None))]
        pub fn from_str_py(contents: &str, format: &str, skip_init: Option<bool>) -> PyResult<Self> {
            SerdeAPI::from_str(contents, format, skip_init.unwrap_or_default()).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Write (serialize) an object to a JSON string
        #[cfg(feature = "json")]
        #[pyo3(name = "to_json")]
        pub fn to_json_py(&self) -> PyResult<String> {
            self.to_json().map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Read (deserialize) an object from a JSON string
        ///
        /// # Arguments
        ///
        /// * `json_str`: `str` - JSON-formatted string to deserialize from
        ///
        #[cfg(feature = "json")]
        #[staticmethod]
        #[pyo3(name = "from_json")]
        #[pyo3(signature = (json_str, skip_init=None))]
        pub fn from_json_py(json_str: &str, skip_init: Option<bool>) -> PyResult<Self> {
            Self::from_json(json_str, skip_init.unwrap_or_default()).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Write (serialize) an object to a message pack
        #[cfg(feature = "msgpack")]
        #[pyo3(name = "to_msg_pack")]
        pub fn to_msg_pack_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
            Ok(PyBytes::new_bound(py, &self.to_msg_pack().map_err(|e| PyIOError::new_err(format!("{:?}", e)))?))
        }

        /// Read (deserialize) an object from a message pack
        ///
        /// # Arguments
        ///
        /// * `msg_pack`: message pack
        ///
        #[cfg(feature = "msgpack")]
        #[staticmethod]
        #[pyo3(name = "from_msg_pack")]
        #[pyo3(signature = (msg_pack, skip_init=None))]
        pub fn from_msg_pack_py(msg_pack: &Bound<PyBytes>, skip_init: Option<bool>) -> PyResult<Self> {
            Self::from_msg_pack(
                msg_pack.as_bytes(),
                skip_init.unwrap_or_default()
            ).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Write (serialize) an object to a TOML string
        #[cfg(feature = "toml")]
        #[pyo3(name = "to_toml")]
        pub fn to_toml_py(&self) -> PyResult<String> {
            self.to_toml().map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Read (deserialize) an object to a TOML string
        ///
        /// # Arguments
        ///
        /// * `toml_str`: `str` - TOML-formatted string to deserialize from
        ///
        #[cfg(feature = "toml")]
        #[staticmethod]
        #[pyo3(name = "from_toml")]
        #[pyo3(signature = (toml_str, skip_init=None))]
        pub fn from_toml_py(toml_str: &str, skip_init: Option<bool>) -> PyResult<Self> {
            Self::from_toml(toml_str, skip_init.unwrap_or_default()).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Write (serialize) an object to a YAML string
        #[cfg(feature = "yaml")]
        #[pyo3(name = "to_yaml")]
        pub fn to_yaml_py(&self) -> PyResult<String> {
            self.to_yaml().map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }

        /// Read (deserialize) an object from a YAML string
        ///
        /// # Arguments
        ///
        /// * `yaml_str`: `str` - YAML-formatted string to deserialize from
        ///
        #[cfg(feature = "yaml")]
        #[staticmethod]
        #[pyo3(name = "from_yaml")]
        #[pyo3(signature = (yaml_str, skip_init=None))]
        pub fn from_yaml_py(yaml_str: &str, skip_init: Option<bool>) -> PyResult<Self> {
            Self::from_yaml(yaml_str, skip_init.unwrap_or_default()).map_err(|e| PyIOError::new_err(format!("{:?}", e)))
        }
    });
}
