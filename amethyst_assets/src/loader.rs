use std::{
    any::{Any, TypeId},
    borrow::Borrow,
    hash::Hash,
    ops::Deref,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use fnv::FnvHashMap;
use log::debug;
use rayon::ThreadPool;

use amethyst_error::{Error as AmethystError, ResultExt};
#[cfg(feature = "profiler")]
use thread_profiler::profile_scope;

use crate::{
    error::Error,
    storage::{AssetStorage, Handle, Processed},
    Asset, Directory, Format, FormatValue, Progress, Source,
};

#[derive(Debug, Clone, Hash)]
struct LoadInfo<'a> {
    path: &'a str,
    source: &'a str,
    type_id: TypeId,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct LoadInfoHash(u64);

impl<'a> From<LoadInfo<'a>> for LoadInfoHash {
    fn from(other: LoadInfo<'a>) -> Self {
        use std::hash::Hasher;

        let mut hasher = fnv::FnvHasher::default();
        other.hash(&mut hasher);
        LoadInfoHash(hasher.finish())
    }
}

/// The asset loader, holding the sources and a reference to the `ThreadPool`.
pub struct Loader {
    hot_reload: bool,
    pool: Arc<ThreadPool>,
    sources: FnvHashMap<String, Arc<dyn Source>>,
    handles: Mutex<FnvHashMap<LoadInfoHash, Box<dyn Any + Send + Sync>>>,
}

impl Loader {
    /// Creates a new asset loader, initializing the directory store with the
    /// given path.
    pub fn new<P>(directory: P, pool: Arc<ThreadPool>) -> Self
    where
        P: Into<PathBuf>,
    {
        Self::with_default_source(Directory::new(directory), pool)
    }

    /// Creates a new asset loader, using the provided source
    pub fn with_default_source<S>(source: S, pool: Arc<ThreadPool>) -> Self
    where
        S: Source,
    {
        let mut loader = Loader {
            hot_reload: true,
            pool,
            sources: Default::default(),
            handles: Default::default(),
        };

        loader.set_default_source(source);
        loader
    }

    /// Add a source to the `Loader`, given an id and the source.
    pub fn add_source<I, S>(&mut self, id: I, source: S)
    where
        I: Into<String>,
        S: Source,
    {
        self.sources
            .insert(id.into(), Arc::new(source) as Arc<dyn Source>);
    }

    /// Set the default source of the `Loader`.
    pub fn set_default_source<S>(&mut self, source: S)
    where
        S: Source,
    {
        self.add_source(String::new(), source);
    }

    /// If set to `true`, this `Loader` will ask formats to
    /// generate "reload instructions" which *allow* reloading.
    /// Calling `set_hot_reload(true)` does not actually enable
    /// hot reloading; this is controlled by the `HotReloadStrategy`
    /// resource.
    pub fn set_hot_reload(&mut self, value: bool) {
        self.hot_reload = value;
    }

    /// Loads an asset with a given format from the default (directory) source.
    /// If you want to load from a custom source instead, use `load_from`.
    ///
    /// See `load_from` for more information.
    pub fn load<A, F, N, P>(
        &self,
        name: N,
        format: F,
        options: F::Options,
        progress: P,
        storage: &AssetStorage<A>,
    ) -> Handle<A>
    where
        A: Asset,
        F: Format<A>,
        N: Into<String>,
        P: Progress,
    {
        #[cfg(feature = "profiler")]
        profile_scope!("initialise_loading_assets");
        self.load_from::<A, F, _, _, _>(name, format, options, "", progress, storage)
    }

    /// Loads an asset with a given format from the default (directory) source.
    /// If you want to load from a custom source instead, use `load_from`.
    ///
    /// See `load_from` for more information.
    pub fn load_or_else<A, F, N, P, EPtr>(
        &self,
        name: N,
        format: F,
        options: F::Options,
        progress: P,
        storage: &AssetStorage<A>,
        or_else: EPtr,
    ) -> Handle<A>
    where
        A: Asset,
        F: Format<A>,
        N: Into<String>,
        P: Progress,
        EPtr: Deref + Send + 'static,
        EPtr::Target: Fn(AmethystError) -> Result<A::Data, AmethystError>,
    {
        #[cfg(feature = "profiler")]
        profile_scope!("initialise_loading_assets");
        self.load_from_or_else::<A, F, _, _, _, _>(
            name, format, options, "", progress, storage, or_else,
        )
    }

    pub fn load_from<A, F, N, P, S>(
        &self,
        name: N,
        format: F,
        options: F::Options,
        source: &S,
        progress: P,
        storage: &AssetStorage<A>,
    ) -> Handle<A>
    where
        A: Asset,
        F: Format<A> + 'static,
        N: Into<String>,
        P: Progress,
        S: AsRef<str> + Eq + Hash + ?Sized,
        String: Borrow<S>,
    {
        self.load_from_or_else(name, format, options, source, progress, storage, &|err| {
            Err(err)
        })
    }

    /// Loads an asset with a given id and format from a custom source.
    /// The actual work is done in a worker thread, thus this method immediately returns a handle.
    ///
    /// ## Parameters
    ///
    /// * `name`: this is just an identifier for the asset, most likely a file name e.g.
    ///   `"meshes/a.obj"`
    /// * `format`: A format struct which loads bytes from a `source` and produces `Asset::Data`
    ///   with them
    /// * `options`: Additional parameter to `format` to configure how exactly the data will
    ///   be created. This could e.g. be mipmap levels for textures.
    /// * `source`: An identifier for a source which has previously been added using `with_source`
    /// * `progress`: A tracker which will be notified of assets which have been imported
    /// * `storage`: The asset storage which can be fetched from the ECS `World` using
    ///   `read_resource`.
    pub fn load_from_or_else<A, F, N, P, S, EPtr>(
        &self,
        name: N,
        format: F,
        options: F::Options,
        source: &S,
        mut progress: P,
        storage: &AssetStorage<A>,
        or_else: EPtr,
    ) -> Handle<A>
    where
        A: Asset,
        F: Format<A> + 'static,
        N: Into<String>,
        P: Progress,
        S: AsRef<str> + Eq + Hash + ?Sized,
        String: Borrow<S>,
        EPtr: Deref + Send + 'static,
        EPtr::Target: Fn(AmethystError) -> Result<A::Data, AmethystError>,
    {
        #[cfg(feature = "profiler")]
        profile_scope!("load_asset_from");
        use crate::progress::Tracker;

        let name = name.into();
        let source = source.as_ref();
        let format_name = F::NAME;
        let source_name = match source {
            "" => "[default source]",
            other => other,
        };

        let key: LoadInfoHash = LoadInfo {
            path: &name,
            source,
            type_id: TypeId::of::<A>(),
        }
        .into();

        let handle = {
            let mut handles = self
                .handles
                .lock()
                .expect("Programmer error: Thread panicked while holding handles lock");

            if let Some(handle) = handles.get(&key) {
                return handle
                    .downcast_ref::<Handle<A>>()
                    .expect("Programmer error: Incorrect type added to map!")
                    .clone();
            }

            let handle = storage.allocate();

            handles.insert(key, Box::new(handle.clone()));

            handle
        };

        debug!(
            "{:?}: Loading asset {:?} with format {:?} from source {:?} (handle id: {:?})",
            A::NAME,
            name,
            format_name,
            source_name,
            handle,
        );

        progress.add_assets(1);
        let tracker = progress.create_tracker();

        let source = self.source(source);
        let handle_clone = handle.clone();
        let processed = storage.processed.clone();

        let hot_reload = self.hot_reload;

        let cl = move || {
            #[cfg(feature = "profiler")]
            profile_scope!("load_asset_from_worker");
            let data = format
                .import(name.clone(), source, options, hot_reload)
                .or_else(|err| or_else(err).map(FormatValue::data))
                .with_context(|_| Error::Format(F::NAME));
            let tracker = Box::new(tracker) as Box<dyn Tracker>;

            processed.push(Processed::NewAsset {
                data,
                handle,
                name,
                tracker,
            });
        };
        self.pool.spawn(cl);

        handle_clone
    }

    /// Load an asset from data and return a handle.
    pub fn load_from_data<A, P>(
        &self,
        data: A::Data,
        mut progress: P,
        storage: &AssetStorage<A>,
    ) -> Handle<A>
    where
        A: Asset,
        P: Progress,
    {
        progress.add_assets(1);
        let tracker = progress.create_tracker();
        let tracker = Box::new(tracker);
        let handle = storage.allocate();
        storage.processed.push(Processed::NewAsset {
            data: Ok(FormatValue::data(data)),
            handle: handle.clone(),
            name: "<Data>".into(),
            tracker,
        });

        handle
    }

    fn source(&self, source: &str) -> Arc<dyn Source> {
        self.sources
            .get(source)
            .expect("No such source. Maybe you forgot to add it with `Loader::add_source`?")
            .clone()
    }
}
