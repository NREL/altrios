use super::cat_power::*;
use super::elev::*;
use super::heading::*;
use super::link_idx::*;
use super::link_old::Link as LinkOld;
use super::speed::*;
use crate::imports::*;

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
struct OldSpeedSets(Vec<OldSpeedSet>);

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, SerdeAPI)]
/// An arbitrary unit of single track that does not include turnouts
#[altrios_api()]
pub struct Link {
    /// Index of current link
    pub idx_curr: LinkIdx,
    /// Index of adjacent link in reverse direction
    pub idx_flip: LinkIdx,
    /// see [EstTime::idx_next]
    pub idx_next: LinkIdx,
    /// see [EstTime::idx_next_alt]  
    /// if it does not exist, it should be `LinkIdx{idx: 0}`
    pub idx_next_alt: LinkIdx,
    /// see [EstTime::idx_prev]
    pub idx_prev: LinkIdx,
    /// see [EstTime::idx_prev_alt]  
    /// if it does not exist, it should be `LinkIdx{idx: 0}`
    pub idx_prev_alt: LinkIdx,

    /// Spatial vector of elevation values and corresponding positions along track
    pub elevs: Vec<Elev>,
    #[serde(default)]
    /// Spatial vector of compass heading values and corresponding positions along track
    pub headings: Vec<Heading>,
    /// Map of train types and corresponding speed sets
    #[serde(default)]
    pub speed_sets: HashMap<TrainType, SpeedSet>,
    /// Optional train-type-neutral [SpeedSet].  If provided, overrides [Link::speed_sets].
    pub speed_set: Option<SpeedSet>,
    #[serde(default)]
    /// Spatial vector of catenary power limit values and corresponding positions along track
    pub cat_power_limits: Vec<CatPowerLimit>,
    pub length: si::Length,

    #[serde(default)]
    pub link_idxs_lockout: Vec<LinkIdx>,
}

impl Link {
    fn is_linked_prev(&self, idx: LinkIdx) -> bool {
        self.idx_curr.is_fake() || self.idx_prev == idx || self.idx_prev_alt == idx
    }
    fn is_linked_next(&self, idx: LinkIdx) -> bool {
        self.idx_curr.is_fake() || self.idx_next == idx || self.idx_next_alt == idx
    }
}

impl From<LinkOld> for Link {
    fn from(l: LinkOld) -> Self {
        let mut speed_sets: HashMap<TrainType, SpeedSet> = HashMap::new();
        for oss in l.speed_sets {
            speed_sets.insert(
                oss.train_type,
                SpeedSet {
                    speed_limits: oss.speed_limits,
                    speed_params: oss.speed_params,
                    is_head_end: oss.is_head_end,
                },
            );
        }

        Self {
            elevs: l.elevs,
            headings: l.headings,
            speed_sets,
            speed_set: Default::default(),
            cat_power_limits: l.cat_power_limits,
            length: l.length,
            idx_next: l.idx_next,
            idx_next_alt: l.idx_next_alt,
            idx_prev: l.idx_prev,
            idx_prev_alt: l.idx_prev_alt,
            idx_curr: l.idx_curr,
            idx_flip: l.idx_flip,
            link_idxs_lockout: l.link_idxs_lockout,
        }
    }
}

impl Valid for Link {
    fn valid() -> Self {
        Self {
            elevs: Vec::<Elev>::valid(),
            headings: Vec::<Heading>::valid(),
            speed_sets: HashMap::<TrainType, SpeedSet>::valid(),
            length: uc::M * 10000.0,
            idx_curr: LinkIdx::valid(),
            ..Self::default()
        }
    }
}

impl ObjState for Link {
    fn is_fake(&self) -> bool {
        self.idx_curr.is_fake()
    }

    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        if self.is_fake() {
            validate_field_fake(&mut errors, &self.idx_next, "Link index next");
            validate_field_fake(&mut errors, &self.idx_next_alt, "Link index next alt");
            validate_field_fake(&mut errors, &self.idx_prev, "Link index prev");
            validate_field_fake(&mut errors, &self.idx_prev_alt, "Link index prev alt");
            validate_field_fake(&mut errors, &self.idx_curr, "Link index curr");
            validate_field_fake(&mut errors, &self.idx_flip, "Link index flip");
            si_chk_num_eqz(&mut errors, &self.length, "Link length");
            validate_field_fake(&mut errors, &self.elevs, "Elevations");
            validate_field_fake(&mut errors, &self.headings, "Headings");
            validate_field_fake(&mut errors, &self.speed_sets, "Speed sets");
            // validate cat_power_limits
            if !self.cat_power_limits.is_empty() {
                errors.push(anyhow!(
                    "Catenary power limits = {:?} must be empty!",
                    self.cat_power_limits
                ));
            }
        } else {
            si_chk_num_gtz(&mut errors, &self.length, "Link length");
            validate_field_real(&mut errors, &self.elevs, "Elevations");
            if !self.headings.is_empty() {
                validate_field_real(&mut errors, &self.headings, "Headings");
            }
            validate_field_real(&mut errors, &self.speed_sets, "Speed sets");
            validate_field_real(&mut errors, &self.cat_power_limits, "Catenary power limits");

            early_err!(errors, "Link");

            if self.idx_flip.is_real() {
                for (var, name) in [
                    (self.idx_curr, "curr"),
                    (self.idx_next, "next"),
                    (self.idx_next_alt, "next alt"),
                    (self.idx_prev, "prev"),
                    (self.idx_prev_alt, "prev alt"),
                ] {
                    if var == self.idx_flip {
                        errors.push(anyhow!(
                            "Link index flip = {:?} and link index {} = {:?} must be different!",
                            self.idx_flip,
                            name,
                            var
                        ));
                    }
                }
            }
            if self.idx_next_alt.is_real() && self.idx_next.is_fake() {
                errors.push(anyhow!(
                    "Link index next = {:?} must be real when link index next alt = {:?} is real!",
                    self.idx_next,
                    self.idx_next_alt
                ));
            }
            if self.idx_prev_alt.is_real() && self.idx_prev.is_fake() {
                errors.push(anyhow!(
                    "Link index prev = {:?} must be real when link index prev alt = {:?} is real!",
                    self.idx_prev,
                    self.idx_prev_alt
                ));
            }

            if self.elevs.first().unwrap().offset != si::Length::ZERO {
                errors.push(anyhow!(
                    "First elevation offset = {:?} is invalid, must equal zero!",
                    self.elevs.first().unwrap().offset
                ));
            }
            if self.elevs.last().unwrap().offset != self.length {
                errors.push(anyhow!(
                    "Last elevation offset = {:?} is invalid, must equal length = {:?}!",
                    self.elevs.last().unwrap().offset,
                    self.length
                ));
            }
            if !self.headings.is_empty() {
                if self.headings.first().unwrap().offset != si::Length::ZERO {
                    errors.push(anyhow!(
                        "First heading offset = {:?} is invalid, must equal zero!",
                        self.headings.first().unwrap().offset
                    ));
                }
                if self.headings.last().unwrap().offset != self.length {
                    errors.push(anyhow!(
                        "Last heading offset = {:?} is invalid, must equal length = {:?}!",
                        self.headings.last().unwrap().offset,
                        self.length
                    ));
                }
            }
            if !self.cat_power_limits.is_empty() {
                if self.cat_power_limits.first().unwrap().offset_start < si::Length::ZERO {
                    errors.push(anyhow!(
                        "First cat power limit offset start = {:?} is invalid, must be greater than or equal to zero!",
                        self.cat_power_limits.first().unwrap().offset_start
                    ));
                }
                if self.cat_power_limits.last().unwrap().offset_end > self.length {
                    errors.push(anyhow!(
                        "Last cat power limit offset end = {:?} is invalid, must be less than or equal to length = {:?}!",
                        self.cat_power_limits.last().unwrap().offset_end,
                        self.length
                    ));
                }
            }
        }
        errors.make_err()
    }
}

#[altrios_api]
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
/// Struct that contains a `Vec<Link>` for the purpose of providing `SerdeAPI` for `Vec<Link>` in
/// Python
pub struct Network(pub Vec<Link>);

impl SerdeAPI for Network {
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
        match Self::from_reader(file, extension) {
            Ok(network) => Ok(network),
            Err(err) => Ok(NetworkOld::from_file(filepath).with_context(|| err)?.into()),
        }
    }
}

impl From<NetworkOld> for Network {
    fn from(value: NetworkOld) -> Self {
        Network(value.0.iter().map(|l| Link::from(l.clone())).collect())
    }
}

#[altrios_api]
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize, SerdeAPI)]
/// Struct that contains a `Vec<Link>` for the purpose of providing `SerdeAPI` for `Vec<Link>` in
/// Python
///
/// # Note:
/// This struct will be deprecated and superseded by [Network]
pub struct NetworkOld(pub Vec<LinkOld>);

impl AsRef<[Link]> for Network {
    fn as_ref(&self) -> &[Link] {
        &self.0
    }
}

impl From<&Vec<Link>> for Network {
    fn from(value: &Vec<Link>) -> Self {
        Self(value.to_vec())
    }
}

#[cfg(feature = "pyo3")]
#[cfg_attr(feature = "pyo3", pyfunction(name = "import_network"))]
pub fn import_network_py(filepath: &PyAny) -> anyhow::Result<Vec<Link>> {
    let network = Vec::<Link>::from_file(PathBuf::extract(filepath)?)?;
    network.validate()?;
    Ok(network)
}

impl Valid for Vec<Link> {
    fn valid() -> Self {
        vec![Link::default(), Link::valid()]
    }
}

impl ObjState for [Link] {
    fn validate(&self) -> ValidationResults {
        let mut errors = ValidationErrors::new();
        if self.len() < 2 {
            errors.push(anyhow!(
                "There must be at least two links (one physical and one dummy)!"
            ));
            return Err(errors);
        }
        validate_slice_fake(&mut errors, &self[..1], "Link");
        validate_slice_real_shift(&mut errors, &self[1..], "Link", 1);
        early_err!(errors, "Links");

        for (idx, link) in self.iter().enumerate().skip(1) {
            // Validate flip and curr
            if link.idx_curr.idx() != idx {
                errors.push(anyhow!(
                    "Link idx {} is not equal to index in vector {}!",
                    link.idx_curr,
                    idx
                ))
            }
            if link.idx_flip == link.idx_curr {
                errors.push(anyhow!(
                    "Normal {} and flipped {} links must be different!",
                    link.idx_curr,
                    link.idx_flip
                ));
            }
            if link.idx_flip.is_real() && self[link.idx_flip.idx()].idx_flip != link.idx_curr {
                errors.push(anyhow!(
                    "Flipped link {} does not properly reference current link {}!",
                    link.idx_flip,
                    link.idx_curr
                ));
            }

            // Validate next
            if link.idx_next.is_real() {
                for (link_next, name) in [
                    (&self[link.idx_next.idx()], "next link"),
                    (&self[link.idx_next_alt.idx()], "next link alt"),
                ] {
                    if !link_next.is_linked_prev(link.idx_curr) {
                        errors.push(anyhow!(
                            "Current link {} with {} {} prev idx {} and prev idx alt {} do not point back!",
                            link.idx_curr,
                            name,
                            link_next.idx_curr,
                            link_next.idx_prev,
                            link_next.idx_prev_alt,
                        ));
                    }
                    if link.idx_next_alt.is_real() && link_next.idx_prev_alt.is_real() {
                        errors.push(anyhow!(
                            "Current link {} and {} {} have coincident switch points!",
                            link.idx_curr,
                            name,
                            link_next.idx_curr,
                        ));
                    }
                }
            } else if link.idx_next_alt.is_real() {
                errors.push(anyhow!(
                    "Next idx alt {} is real when next idx {} is fake!",
                    link.idx_next_alt,
                    link.idx_next,
                ));
            }

            // Validate prev
            if link.idx_prev.is_real() {
                for (link_prev, name) in [
                    (&self[link.idx_prev.idx()], "prev link"),
                    (&self[link.idx_prev_alt.idx()], "prev link alt"),
                ] {
                    if !link_prev.is_linked_next(link.idx_curr) {
                        errors.push(anyhow!(
                            "Current link {} with {} {} next idx {} and next idx alt {} do not point back!",
                            link.idx_curr,
                            name,
                            link_prev.idx_curr,
                            link_prev.idx_next,
                            link_prev.idx_next_alt,
                        ));
                    }
                    if link.idx_prev_alt.is_real() && link_prev.idx_next_alt.is_real() {
                        errors.push(anyhow!(
                            "Current link {} and {} {} have coincident switch points!",
                            link.idx_curr,
                            name,
                            link_prev.idx_curr,
                        ));
                    }
                }
            } else if link.idx_prev_alt.is_real() {
                errors.push(anyhow!(
                    "Prev idx alt {} is real when prev idx {} is fake!",
                    link.idx_prev_alt,
                    link.idx_prev
                ));
            }
        }
        errors.make_err()
    }
}

#[cfg(test)]
mod test_link {
    use super::*;
    use crate::testing::*;

    impl Cases for Link {
        fn real_cases() -> Vec<Self> {
            vec![
                Self::valid(),
                Self {
                    idx_flip: LinkIdx::new(2),
                    ..Self::valid()
                },
                Self {
                    idx_next: Self::valid().idx_curr,
                    ..Self::valid()
                },
                Self {
                    idx_prev: Self::valid().idx_curr,
                    ..Self::valid()
                },
                Self {
                    idx_next: LinkIdx::new(2),
                    idx_next_alt: Self::valid().idx_curr,
                    ..Self::valid()
                },
                Self {
                    idx_prev: LinkIdx::new(2),
                    idx_prev_alt: Self::valid().idx_curr,
                    ..Self::valid()
                },
            ]
        }
        fn fake_cases() -> Vec<Self> {
            vec![Self::default()]
        }
        fn invalid_cases() -> Vec<Self> {
            vec![
                Self {
                    elevs: Vec::<Elev>::invalid_cases().first().unwrap().to_vec(),
                    ..Self::valid()
                },
                Self {
                    elevs: Vec::<Elev>::new(),
                    ..Self::valid()
                },
                Self {
                    length: si::Length::ZERO,
                    ..Self::valid()
                },
                Self {
                    length: -uc::M,
                    ..Self::valid()
                },
                Self {
                    length: uc::M * f64::NAN,
                    ..Self::valid()
                },
                Self {
                    idx_curr: LinkIdx::default(),
                    ..Self::valid()
                },
                Self {
                    idx_flip: Self::valid().idx_curr,
                    ..Self::valid()
                },
                Self {
                    idx_next_alt: Self::valid().idx_curr,
                    ..Self::valid()
                },
                Self {
                    idx_prev_alt: Self::valid().idx_curr,
                    ..Self::valid()
                },
                Self {
                    idx_flip: LinkIdx::new(2),
                    idx_next: LinkIdx::new(2),
                    ..Self::valid()
                },
                Self {
                    idx_flip: LinkIdx::new(2),
                    idx_next: LinkIdx::new(3),
                    idx_next_alt: LinkIdx::new(2),
                    ..Self::valid()
                },
                Self {
                    idx_flip: LinkIdx::new(2),
                    idx_prev: LinkIdx::new(2),
                    ..Self::valid()
                },
                Self {
                    idx_flip: LinkIdx::new(2),
                    idx_prev: LinkIdx::new(3),
                    idx_prev_alt: LinkIdx::new(2),
                    ..Self::valid()
                },
            ]
        }
    }

    check_cases!(Link);

    #[test]
    fn check_elevs_start() {
        for mut link in Link::real_cases() {
            link.elevs.first_mut().unwrap().offset -= uc::M;
            link.validate().unwrap_err();
        }
    }

    #[test]
    fn check_elevs_end() {
        for mut link in Link::real_cases() {
            link.elevs.last_mut().unwrap().offset += uc::M;
            link.validate().unwrap_err();
        }
    }
}

#[cfg(test)]
mod test_links {
    use super::*;
    use crate::testing::*;

    impl Cases for Vec<Link> {
        fn real_cases() -> Vec<Self> {
            vec![Self::valid()]
        }
        fn invalid_cases() -> Vec<Self> {
            vec![vec![], Self::valid()[..1].to_vec()]
        }
    }
    //check_cases!(Vec<Link>);
    //check_vec_elems!(Link);

    #[test]
    fn test_to_and_from_file_for_links() {
        let links = Vec::<Link>::valid();
        links.to_file("links_test2.yaml").unwrap();
        assert_eq!(Vec::<Link>::from_file("links_test2.yaml").unwrap(), links);
        std::fs::remove_file("links_test2.yaml").unwrap();
    }
}
