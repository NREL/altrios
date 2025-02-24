use crate::imports::*;

///Standardizes conversion from smaller than usize types for indexing.
pub trait Idx {
    fn idx(self) -> usize;
}

#[duplicate_item(Self; [u8]; [u16])]
impl Idx for Self {
    fn idx(self) -> usize {
        self.into()
    }
}

impl Idx for u32 {
    fn idx(self) -> usize {
        self.try_into().unwrap()
    }
}

impl Idx for Option<NonZeroU16> {
    fn idx(self) -> usize {
        self.map(u16::from).unwrap_or(0) as usize
    }
}

/// Trait implemented for indexing types, specifically `usize`, to assist in
/// converting them into an `Option<NonZeroUxxxx>`.
///
/// This is necessary because both the common type conversion trait ([From])
/// and the types involved (e.g. `Option<NonZeroU16>` ) are from the
/// standard library.  Rust does not allow for the combination of traits and
/// types if both are from external libraries.
///
/// This trait is default implemented for [usize] to convert into any type that
/// already implements [TryFrom]<[NonZeroUsize]>, which includes most of the
/// NonZeroUxxxx types.
///
/// This approach will error on a loss of precision.  So, if the [usize] value
/// does not fit into the `NonZero` type, then an [Error] variant is returned.
///
/// If the value is `0`, then an [Ok]\([None]) variant is returned.
///
/// Note that the base `NonZero` types already have a similar feature if
/// converting from the same basic type (e.g. [u16] to [Option]<[NonZeroU16]>),
/// where the type is guaranteed to fit, but just might be `0`.  If that is the
/// use case, then just use the `new()` method, which returns the [None] variant
/// if the value is `0`.
///
/// The intended usage is as follows:
/// ```
/// # use std::num::NonZeroU8;
/// # use altrios_core::traits::TryFromIdx;
///
/// let non_zero : usize = 42;
/// let good_val : Option<NonZeroU8> = non_zero.try_from_idx().unwrap();
/// assert!(good_val == NonZeroU8::new(42));
///
/// let zero : usize = 0;
/// let none_val : Option<NonZeroU8> = zero.try_from_idx().unwrap();
/// assert!(none_val == None);
///
/// let too_big : usize = 256;
/// let bad_val : Result<Option<NonZeroU8>, _> = too_big.try_from_idx();
/// assert!(bad_val.is_err());
/// ```
pub trait TryFromIdx<T> {
    type Error;

    fn try_from_idx(&self) -> Result<Option<T>, Self::Error>;
}

impl<T> TryFromIdx<T> for usize
where
    T: TryFrom<NonZeroUsize>,
{
    type Error = <T as TryFrom<NonZeroUsize>>::Error;

    fn try_from_idx(&self) -> Result<Option<T>, Self::Error> {
        NonZeroUsize::new(*self).map_or(
            // If value is a 0-valued usize, then we immediately return an
            // Ok(None).
            Ok(None),
            // Otherwise we attempt to convert it from a NonZeroUsize into a
            // different type with potentially smaller accuracy.
            |val| {
                T::try_from(val)
                    // We wrap a valid result in Some
                    .map(Some)
            },
        )
    }
}

pub trait Linspace {
    fn linspace(start: f64, stop: f64, n_elements: usize) -> Vec<f64> {
        let n_steps = n_elements - 1;
        let step_size = (stop - start) / n_steps as f64;
        let v_norm: Vec<f64> = (0..=n_steps)
            .collect::<Vec<usize>>()
            .iter()
            .map(|x| *x as f64)
            .collect();
        let v = v_norm.iter().map(|x| (x * step_size) + start).collect();
        v
    }
}

impl Linspace for Vec<f64> {}

pub trait SerdeAPI: Serialize + for<'a> Deserialize<'a> {
    const ACCEPTED_BYTE_FORMATS: &'static [&'static str] = &["yaml", "json", "bin"];
    const ACCEPTED_STR_FORMATS: &'static [&'static str] = &["yaml", "json"];

    /// Specialized code to execute upon initialization
    fn init(&mut self) -> Result<(), Error> {
        Ok(())
    }

    /// Write (serialize) an object to a file.
    /// Supported file extensions are listed in [`ACCEPTED_BYTE_FORMATS`](`SerdeAPI::ACCEPTED_BYTE_FORMATS`).
    /// Creates a new file if it does not already exist, otherwise truncates the existing file.
    ///
    /// # Arguments
    ///
    /// * `filepath` - The filepath at which to write the object
    ///
    fn to_file<P: AsRef<Path>>(&self, filepath: P) -> anyhow::Result<()> {
        let filepath = filepath.as_ref();
        let extension = filepath
            .extension()
            .and_then(OsStr::to_str)
            .with_context(|| format!("File extension could not be parsed: {filepath:?}"))?;
        match extension.trim_start_matches('.').to_lowercase().as_str() {
            "yaml" | "yml" => serde_yaml::to_writer(&File::create(filepath)?, self)?,
            "json" => serde_json::to_writer(&File::create(filepath)?, self)?,
            "bin" => bincode::serialize_into(&File::create(filepath)?, self)?,
            _ => bail!(
                "Unsupported format {extension:?}, must be one of {:?}",
                Self::ACCEPTED_BYTE_FORMATS
            ),
        }
        Ok(())
    }

    /// Read (deserialize) an object from a file.
    /// Supported file extensions are listed in [`ACCEPTED_BYTE_FORMATS`](`SerdeAPI::ACCEPTED_BYTE_FORMATS`).
    ///
    /// # Arguments:
    ///
    /// * `filepath`: The filepath from which to read the object
    ///
    fn from_file<P: AsRef<Path>>(filepath: P, skip_init: bool) -> anyhow::Result<Self> {
        let filepath = filepath.as_ref();
        let extension = filepath
            .extension()
            .and_then(OsStr::to_str)
            .with_context(|| format!("File extension could not be parsed: {filepath:?}"))?;
        let mut file = File::open(filepath).with_context(|| {
            if !filepath.exists() {
                format!("File not found: {filepath:?}")
            } else {
                format!("Could not open file: {filepath:?}")
            }
        })?;
        Self::from_reader(&mut file, extension, skip_init)
    }

    /// Write (serialize) an object into a string
    ///
    /// # Arguments:
    ///
    /// * `format` - The target format, any of those listed in [`ACCEPTED_STR_FORMATS`](`SerdeAPI::ACCEPTED_STR_FORMATS`)
    ///
    fn to_str(&self, format: &str) -> anyhow::Result<String> {
        match format.trim_start_matches('.').to_lowercase().as_str() {
            "yaml" | "yml" => self.to_yaml(),
            "json" => self.to_json(),
            _ => bail!(
                "Unsupported format {format:?}, must be one of {:?}",
                Self::ACCEPTED_STR_FORMATS
            ),
        }
    }

    /// Read (deserialize) an object from a string
    ///
    /// # Arguments:
    ///
    /// * `contents` - The string containing the object data
    /// * `format` - The source format, any of those listed in [`ACCEPTED_STR_FORMATS`](`SerdeAPI::ACCEPTED_STR_FORMATS`)
    ///
    fn from_str<S: AsRef<str>>(contents: S, format: &str, skip_init: bool) -> anyhow::Result<Self> {
        Ok(
            match format.trim_start_matches('.').to_lowercase().as_str() {
                "yaml" | "yml" => Self::from_yaml(contents, skip_init)?,
                "json" => Self::from_json(contents, skip_init)?,
                _ => bail!(
                    "Unsupported format {format:?}, must be one of {:?}",
                    Self::ACCEPTED_STR_FORMATS
                ),
            },
        )
    }

    /// Deserialize an object from anything that implements [`std::io::Read`]
    ///
    /// # Arguments:
    ///
    /// * `rdr` - The reader from which to read object data
    /// * `format` - The source format, any of those listed in [`ACCEPTED_BYTE_FORMATS`](`SerdeAPI::ACCEPTED_BYTE_FORMATS`)
    ///
    fn from_reader<R: std::io::Read>(
        rdr: &mut R,
        format: &str,
        skip_init: bool,
    ) -> anyhow::Result<Self> {
        let mut deserialized: Self = match format.trim_start_matches('.').to_lowercase().as_str() {
            "yaml" | "yml" => serde_yaml::from_reader(rdr)?,
            "json" => serde_json::from_reader(rdr)?,
            #[cfg(feature = "msgpack")]
            "msgpack" => rmp_serde::decode::from_read(rdr)?,
            _ => bail!(
                "Unsupported format {format:?}, must be one of {:?}",
                Self::ACCEPTED_BYTE_FORMATS
            ),
        };
        if !skip_init {
            deserialized.init()?;
        }
        Ok(deserialized)
    }

    /// Write (serialize) an object to a JSON string
    fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string(&self)?)
    }

    /// Read (deserialize) an object from a JSON string
    ///
    /// # Arguments
    ///
    /// * `json_str` - JSON-formatted string to deserialize from
    ///
    fn from_json<S: AsRef<str>>(json_str: S, skip_init: bool) -> anyhow::Result<Self> {
        let mut json_de: Self = serde_json::from_str(json_str.as_ref())?;
        if !skip_init {
            json_de.init()?;
        }
        Ok(json_de)
    }

    /// Write (serialize) an object to a YAML string
    fn to_yaml(&self) -> anyhow::Result<String> {
        Ok(serde_yaml::to_string(&self)?)
    }

    /// Write (serialize) an object to a message pack
    #[cfg(feature = "msgpack")]
    fn to_msg_pack(&self) -> anyhow::Result<Vec<u8>> {
        Ok(rmp_serde::encode::to_vec_named(&self)?)
    }

    /// Read (deserialize) an object from a message pack
    ///
    /// # Arguments
    ///
    /// * `msg_pack` - message pack object
    ///
    #[cfg(feature = "msgpack")]
    fn from_msg_pack(msg_pack: &[u8], skip_init: bool) -> anyhow::Result<Self> {
        let mut msg_pack_de: Self = rmp_serde::decode::from_slice(msg_pack)?;
        if !skip_init {
            msg_pack_de.init()?;
        }
        Ok(msg_pack_de)
    }

    /// Read (deserialize) an object from a YAML string
    ///
    /// # Arguments
    ///
    /// * `yaml_str` - YAML-formatted string to deserialize from
    ///
    fn from_yaml<S: AsRef<str>>(yaml_str: S, skip_init: bool) -> anyhow::Result<Self> {
        let mut yaml_de: Self = serde_yaml::from_str(yaml_str.as_ref())?;
        if !skip_init {
            yaml_de.init()?;
        }
        Ok(yaml_de)
    }

    /// Write (serialize) an object to bincode-encoded bytes
    fn to_bincode(&self) -> anyhow::Result<Vec<u8>> {
        Ok(bincode::serialize(&self)?)
    }

    /// Read (deserialize) an object from bincode-encoded bytes
    ///
    /// # Arguments
    ///
    /// * `encoded` - Encoded bytes to deserialize from
    ///
    fn from_bincode(encoded: &[u8]) -> anyhow::Result<Self> {
        let mut bincode_de: Self = bincode::deserialize(encoded)?;
        bincode_de.init()?;
        Ok(bincode_de)
    }
}

impl<T: SerdeAPI> SerdeAPI for Vec<T> {
    fn init(&mut self) -> Result<(), Error> {
        for val in self {
            val.init()?
        }
        Ok(())
    }
}

/// Provides method for checking if an instance of `Self` is equal to `Self::default`
pub trait EqDefault: Default + PartialEq {
    /// Checks if an instance of `Self` is equal to `Self::default`
    fn eq_default(&self) -> bool {
        *self == Self::default()
    }
}
impl<T: Default + PartialEq> EqDefault for T {}

#[derive(Default, Deserialize, Serialize, Debug, Clone, PartialEq)]
/// Governs which side effect to trigger when setting mass
pub enum MassSideEffect {
    /// To be used when [MassSideEffect] is not applicable
    #[default]
    None,
    /// Set the extensive parameter -- e.g. energy, power -- as a side effect
    Extensive,
    /// Set the intensive parameter -- e.g. specific power, specific energy -- as a side effect
    Intensive,
}

impl TryFrom<String> for MassSideEffect {
    type Error = anyhow::Error;
    fn try_from(value: String) -> anyhow::Result<MassSideEffect> {
        let mass_side_effect = match value.as_str() {
            "None" => Self::None,
            "Extensive" => Self::Extensive,
            "Intensive" => Self::Intensive,
            _ => {
                bail!(format!(
                    "`MassSideEffect` must be 'Intensive', 'Extensive', or 'None'. "
                ))
            }
        };
        Ok(mass_side_effect)
    }
}

pub trait Mass {
    /// Returns mass of Self, either from `self.mass` or
    /// the derived from fields that store mass data. `Mass::mass` also checks that
    /// derived mass, if Some, is same as `self.mass`.
    fn mass(&self) -> anyhow::Result<Option<si::Mass>>;

    /// Sets component mass to `mass`, or if `None` is provided for `mass`,
    /// sets mass based on other component parameters (e.g. power and power
    /// density, sum of fields containing mass)
    fn set_mass(
        &mut self,
        new_mass: Option<si::Mass>,
        side_effect: MassSideEffect,
    ) -> anyhow::Result<()>;

    /// Returns derived mass (e.g. sum of mass fields, or
    /// calculation involving mass specific properties).  If
    fn derived_mass(&self) -> anyhow::Result<Option<si::Mass>>;

    /// Sets all fields that are used in calculating derived mass to `None`.
    /// Does not touch `self.mass`.
    fn expunge_mass_fields(&mut self);

    /// Sets any mass-specific property with appropriate side effects
    fn set_mass_specific_property(&mut self) -> anyhow::Result<()> {
        // TODO: remove this default implementation when this method has been universally implemented.
        // For structs without specific properties, return an error
        Ok(())
    }
}
