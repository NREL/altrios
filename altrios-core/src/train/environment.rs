use super::train_imports::*;

#[altrios_api(
    #[staticmethod]
    #[pyo3(name = "from_csv_file")]
    fn from_csv_file_py(pathstr: String) -> anyhow::Result<Self> {
        Self::from_csv_file(&pathstr)
    }

    fn __len__(&self) -> usize {
        self.len()
    }
)]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, SerdeAPI)]
/// Container
pub struct TemperatureTrace {
    /// simulation elapsed time
    pub time: Vec<si::Time>,
    /// ambient temperature at sea level
    pub temp_at_sea_level: Vec<si::ThermodynamicTemperature>,
}

impl TemperatureTrace {
    pub fn empty() -> Self {
        Self {
            time: Vec::new(),
            temp_at_sea_level: Vec::new(),
        }
    }

    pub fn dt(&self, i: usize) -> si::Time {
        self.time[i] - self.time[i - 1]
    }

    pub fn len(&self) -> usize {
        self.time.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn push(&mut self, tt_element: TemperatureTraceElement) {
        self.time.push(tt_element.time);
        self.temp_at_sea_level.push(tt_element.temp_at_sea_level);
    }

    pub fn trim(&mut self, start_idx: Option<usize>, end_idx: Option<usize>) -> anyhow::Result<()> {
        let start_idx = start_idx.unwrap_or(0);
        let end_idx = end_idx.unwrap_or_else(|| self.len());
        ensure!(end_idx <= self.len(), format_dbg!(end_idx <= self.len()));

        self.time = self.time[start_idx..end_idx].to_vec();
        Ok(())
    }

    /// Load cycle from csv file
    pub fn from_csv_file(pathstr: &str) -> Result<Self, anyhow::Error> {
        let pathbuf = PathBuf::from(&pathstr);

        // create empty temperature trace to be populated
        let mut tt = Self::empty();

        let file = File::open(pathbuf)?;
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);
        for result in rdr.deserialize() {
            let tt_elem: TemperatureTraceElement = result?;
            tt.push(tt_elem);
        }
        if tt.is_empty() {
            bail!("Invalid TemperatureTrace file; TemperatureTrace is empty")
        } else {
            Ok(tt)
        }
    }
}

impl Default for TemperatureTrace {
    fn default() -> Self {
        let time_s: Vec<f64> = (0..1).map(|x| x as f64).collect();
        let mut tt = Self {
            time: time_s.iter().map(|t| *t * uc::S).collect(),
            temp_at_sea_level: vec![(22.0 + 273.15) * uc::KELVIN],
        };
        tt.init().unwrap();
        tt
    }
}

/// Element of [TemperatureTrace].  Used for vec-like operations.
#[derive(Default, Debug, Serialize, Deserialize, PartialEq, SerdeAPI)]
pub struct TemperatureTraceElement {
    /// simulation time
    time: si::Time,
    /// ambient temperature at sea level
    pub temp_at_sea_level: si::ThermodynamicTemperature,
}
