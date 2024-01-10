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

    /// Runs any initialization steps that might be needed
    fn init(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    /// Save current data structure to file. Method adaptively calls serialization methods
    /// dependent on the suffix of the filepath.
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

    /// Read from file and return instantiated struct. Method adaptively calls deserialization
    /// methods dependent on the suffix of the file name given as str.
    /// Function returns a dynamic Error Result if it fails.
    ///
    /// # Argument:
    ///
    /// * `filepath`: a `str` storing the targeted file name. Currently `.json` and `.yaml` suffixes are
    /// supported
    ///
    /// # Returns:
    ///
    /// A Rust Result wrapping data structure if method is called successfully; otherwise a dynamic
    /// Error.
    fn from_file<P: AsRef<Path>>(filepath: P) -> anyhow::Result<Self> {
        let filepath = filepath.as_ref();
        let extension = filepath
            .extension()
            .and_then(OsStr::to_str)
            .with_context(|| format!("File extension could not be parsed: {filepath:?}"))?;
        let file = File::open(filepath).with_context(|| {
            if !filepath.exists() {
                format!("File not found: {filepath:?}")
            } else {
                format!("Could not open file: {filepath:?}")
            }
        })?;
        // deserialized file
        let mut deserialized = Self::from_reader(file, extension)?;
        deserialized.init()?;
        Ok(deserialized)
    }

    fn from_reader<R: std::io::Read>(rdr: R, format: &str) -> anyhow::Result<Self> {
        Ok(
            match format.trim_start_matches('.').to_lowercase().as_str() {
                "yaml" | "yml" => serde_yaml::from_reader(rdr)?,
                "json" => serde_json::from_reader(rdr)?,
                "bin" => bincode::deserialize_from(rdr)?,
                _ => bail!(
                    "Unsupported format {format:?}, must be one of {:?}",
                    Self::ACCEPTED_BYTE_FORMATS
                ),
            },
        )
    }

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

    fn from_str(contents: &str, format: &str) -> anyhow::Result<Self> {
        match format.trim_start_matches('.').to_lowercase().as_str() {
            "yaml" | "yml" => Self::from_yaml(contents),
            "json" => Self::from_json(contents),
            _ => bail!(
                "Unsupported format {format:?}, must be one of {:?}",
                Self::ACCEPTED_STR_FORMATS
            ),
        }
    }

    /// JSON serialization method
    fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string(&self)?)
    }

    /// JSON deserialization method
    fn from_json(json_str: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json_str)?)
    }

    /// YAML serialization method
    fn to_yaml(&self) -> anyhow::Result<String> {
        Ok(serde_yaml::to_string(&self)?)
    }

    /// YAML deserialization method
    fn from_yaml(yaml_str: &str) -> anyhow::Result<Self> {
        Ok(serde_yaml::from_str(yaml_str)?)
    }

    /// bincode serialization method
    fn to_bincode(&self) -> anyhow::Result<Vec<u8>> {
        Ok(bincode::serialize(&self)?)
    }

    /// bincode deserialization method
    fn from_bincode(encoded: &[u8]) -> anyhow::Result<Self> {
        Ok(bincode::deserialize(encoded)?)
    }
}

impl<T: SerdeAPI> SerdeAPI for Vec<T> {
    fn init(&mut self) -> anyhow::Result<()> {
        for val in self {
            val.init()?
        }
        Ok(())
    }
}
