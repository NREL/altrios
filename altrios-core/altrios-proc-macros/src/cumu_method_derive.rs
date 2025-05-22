use crate::imports::*;

lazy_static! {
    static ref ENERGY_REGEX: Regex = Regex::new(r"energy_(\w+)").unwrap();
}

pub(crate) fn cumu_method_derive(input: TokenStream) -> TokenStream {
    let item_struct = syn::parse_macro_input!(input as syn::ItemStruct);
    let ident = &item_struct.ident;
    let mut impl_block = TokenStream2::default();

    let fields = if let syn::Fields::Named(syn::FieldsNamed { named, .. }) = item_struct.fields {
        named
    } else {
        abort_call_site!("`SetCumulative` works only on Named Field structs.")
    };

    let struct_has_state = fields.iter().any(|x| *x.ident.as_ref().unwrap() == "state");
    let ident_str = ident.to_string();
    let struct_is_state = ident_str.contains("State");
    let fields_with_state_vec: Vec<bool> = fields
        .iter()
        .map(|field| {
            field
                .attrs
                .iter()
                .any(|attr| attr.path().is_ident("has_state"))
        })
        .collect();

    let (pwr_fields, energy_fields): (Vec<TokenStream2>, Vec<TokenStream2>) = if struct_is_state {
        fields
            .iter()
            .filter_map(|x| {
                let field_str = &x.ident.as_ref().unwrap().to_string();
                if ENERGY_REGEX.is_match(field_str) {
                    let key = ENERGY_REGEX.captures(field_str).unwrap()[1].to_string();
                    if fields
                        .iter()
                        .any(|x| *x.ident.as_ref().unwrap() == format!("pwr_{}", key))
                    {
                        Some((
                            format!("pwr_{}", key).parse().unwrap(),
                            field_str.clone().parse().unwrap(),
                        ))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .unzip()
    } else {
        (Default::default(), Default::default())
    };

    // fields that contain nested `state` fields
    let fields_with_state = fields
        .iter()
        .zip(fields_with_state_vec)
        .filter(|(_f, hsv)| *hsv)
        .map(|(f, _hsv)| f.ident.as_ref().unwrap())
        .collect::<Vec<_>>();

    if struct_is_state {
        impl_block.extend::<TokenStream2>(quote! {
            // this tells the compiler that the `SetCumulative` trait is not manually derived
            #[automatically_derived]
            impl SetCumulative for #ident {
                fn set_cumulative<F: Fn() -> String>(&mut self, dt: si::Time, loc: F) -> anyhow::Result<()> {
                    #(self
                        .#energy_fields
                        .increment(
                            *self.#pwr_fields.get_fresh(|| format_dbg!())? * dt,
                            || format_dbg!()
                        )?;
                    )*
                    Ok(())
                }
            }
        });
    } else if struct_has_state {
        impl_block.extend::<TokenStream2>(quote! {
            // this tells the compiler that the `SetCumulative` trait is not manually derived
            #[automatically_derived]
            impl SetCumulative for #ident {
                fn set_cumulative<F: Fn() -> String>(&mut self, dt: si::Time, loc: F) -> anyhow::Result<()> {
                    self.state.set_cumulative(dt, || format_dbg!())?;
                    #(self.#fields_with_state.set_cumulative(dt, || format_dbg!())?;)*
                    Ok(())
                }
            }
        });
    } else {
        impl_block.extend::<TokenStream2>(quote! {
            // this tells the compiler that the `SetCumulative` trait is not manually derived
            #[automatically_derived]
            impl SetCumulative for #ident {
                fn set_cumulative<F: Fn() -> String>(&mut self, dt: si::Time, loc: F) -> anyhow::Result<()> {
                    #(self.#fields_with_state.set_cumulative(dt, || format_dbg!())?;)*
                    Ok(())
                }
            }
        });
    }

    impl_block.into()
}
