pub mod io {
    use ndarray::{Array2, ArrayView1};
    use num_traits;

    /// Read csv file into Array2
    ///
    /// # Arguments
    ///
    /// * `fname` - filename
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::Array2
    /// let values: Array2<f64> = String::from("values.csv");
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if parsing `T` from the string in a file fails
    pub fn read_csv<T>(fname: String) -> Array2<T>
    where
        T: Clone + num_traits::identities::Zero + std::str::FromStr + std::fmt::Debug,
        T::Err: std::fmt::Debug,
    {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .delimiter(b';')
            .from_path(fname)
            .unwrap();

        let cols = reader.headers().unwrap().len();
        let rows = reader.records().count();

        println!("Reading csv...");
        println!("Found {} columns and {} rows.", rows, cols);

        // move to beginning of file
        let pos = csv::Position::new();
        reader.seek(pos).unwrap();

        let mut out_vals = Array2::<T>::zeros((rows, cols));
        for (row_idx, (record, mut out_row)) in reader
            .records()
            .skip(1)
            .zip(out_vals.outer_iter_mut())
            .enumerate()
        {
            let cur_record: csv::StringRecord = record.unwrap();
            for (col_idx, (out_field, record_field)) in
                out_row.iter_mut().zip(cur_record.iter()).enumerate()
            {
                *out_field = match record_field.parse() {
                    Result::Err(err) => panic!(
                        "Error tring to parse value in ({}, {})!\n Error: {:?}",
                        row_idx, col_idx, err
                    ),
                    Result::Ok(val) => val,
                };
            }
        }
        out_vals
    }

    /// Convert values from ndarray::Array1 to csv::StringRecord
    ///
    /// # Arguments
    ///
    /// * `vals` - values to convert
    fn to_record<T>(vals: &ArrayView1<T>) -> csv::StringRecord
    where
        T: std::fmt::Display,
    {
        let mut record = csv::StringRecord::with_capacity(10, vals.dim());
        for val in vals.iter() {
            record.push_field(&format!("{}", val).as_str());
        }
        record
    }

    /// Write ndarray::Array2 to csv file
    ///
    /// # Arguments
    ///
    /// * `arr` - array to write to file
    /// * `fname` - target filename
    /// * `delimiter' - delimiter to use in csv
    pub fn to_csv<T>(arr: Array2<T>, fname: String, delimiter: u8)
    where
        T: std::fmt::Display,
    {
        let mut writer = csv::WriterBuilder::new()
            .delimiter(delimiter)
            .from_path(fname)
            .unwrap();

        for row in arr.outer_iter() {
            writer.write_record(&to_record(&row)).unwrap();
        }
        writer.flush().unwrap();
    }
}

pub mod algo {
    use ndarray::{self, Array1, Array2, ArrayView1, Axis};
    use std::ops::{Div, Mul, Sub};

    /// Compute Squared distance between 2 Arrays / Points of Data
    ///
    /// # Arguments
    ///
    /// * `a`, `b` - Arrays to compute distances between
    fn dist_sq<T>(a: ArrayView1<T>, b: ArrayView1<T>) -> f64
    where
        T: Clone + Copy + Mul<Output = T> + Sub<Output = T> + num_traits::Zero,
        f64: From<T>,
        Array1<T>: Sub<Output = Array1<T>>,
    {
        f64::from(
            (a.into_owned() - b.into_owned())
                .mapv(|val: T| -> T { val * val })
                .sum(),
        )
    }

    /// Calculate fuzzy memberships for elements of data to clusters in cluster and write to memberships
    ///
    /// # Arguments
    ///
    /// * `q`            - fuzzifier
    /// * `data`         - data to compute memberships for
    /// * `clusters`     - clusters
    /// * `memberships`  - write membership information here
    pub fn compute_memberships<T>(
        q: f64,
        data: &Array2<T>,
        clusters: &Array2<T>,
        memberships: &mut Array2<f64>,
    ) where
        T: Clone + Copy + Mul<Output = T> + Sub<Output = T> + num_traits::Zero + Div<Output = T>,
        f64: From<T>,
        Array1<T>: Sub<Output = Array1<T>>,
    {
        // Membrships are distances for now
        for (i, val) in data.outer_iter().enumerate() {
            for (j, cluster) in clusters.outer_iter().enumerate() {
                memberships[[i, j]] = dist_sq(val, cluster).powf(1.0 / (1.0 - q));
            }
        }

        // compute cluster memberships
        let dist_sums = memberships.sum_axis(Axis(1));
        for (mut cluster_dists, dist_sum) in memberships.outer_iter_mut().zip(dist_sums.iter()) {
            cluster_dists.mapv_inplace(|val| val / *dist_sum);
        }
    }

    /// Compute nearest cluster per data point from clusters
    ///
    /// # Arguments
    ///
    /// `data` - datapoints to compute memberships for.
    /// `clusters` - Cluster Centers to compute nearest cluster for
    pub fn compute_nearest<T>(data: &Array2<T>, clusters: &Array2<T>) -> Array2<T>
    where
        T: Clone
            + Copy
            + Mul<Output = T>
            + Sub<Output = T>
            + PartialOrd
            + num_traits::Zero
            + num_traits::Pow<f64, Output = T>
            + Div<Output = T>
            + std::convert::From<i32>,
        f64: From<T>,
        Array1<T>: Sub<Output = Array1<T>>,
    {
        let mut out = data.clone();
        let mut nearest_clusters = Array1::<T>::zeros(data.dim().0);

        for (val, nearest) in data.outer_iter().zip(nearest_clusters.iter_mut()) {
            let mut min_cluster: i32 = 0;

            let dists = clusters.map_axis(Axis(1), |cluster| dist_sq(val, cluster));
            let mut min_dist = dists[0];
            for (j, cluster) in clusters.outer_iter().enumerate().skip(1) {
                let cur_dist = dist_sq(val, cluster);
                if cur_dist < min_dist {
                    min_dist = cur_dist;
                    min_cluster = j as i32;
                }
            }
            *nearest = T::from(min_cluster);
        }

        out.push_column(nearest_clusters.view()).unwrap();
        out
    }

    /// Compute cluster means using fuzzy k means clustering
    ///
    /// # Arguments
    /// `k` - number of clusters
    /// `n_iter` - number of iterations to perform
    /// `q` - fuzzifier
    /// `data` - data to cluster (rows are data points)
    pub fn cluster_k_means_fuzzy<T>(k: usize, n_iter: usize, q: f64, data: &Array2<T>) -> Array2<T>
    where
        T: Clone
            + Copy
            + Mul<Output = T>
            + Sub<Output = T>
            + PartialOrd
            + num_traits::Zero
            + num_traits::Pow<f64, Output = T>
            + Div<Output = T>
            + std::convert::From<i32>
            + std::convert::From<f64>,
        f64: From<T>,
        rand::distributions::Standard: rand::prelude::Distribution<T>,
        Array1<T>: Sub<Output = Array1<T>>,
    {
        let size = data.dim();

        // cluster initialisation as random between 0 and 1
        let mut clusters = Array2::<T>::zeros((k, size.1));
        for mut cluster in clusters.outer_iter_mut() {
            for i in 0..size.1 {
                cluster[i] = rand::random();
            }
        }

        for _ in 0..n_iter {
            let mut memberships = Array2::<f64>::zeros((size.0, k));
            compute_memberships(q, &data, &clusters, &mut memberships);

            // compute new cluster means
            for (mut cluster, membership) in clusters
                .axis_iter_mut(Axis(0))
                .zip(memberships.axis_iter(Axis(1)))
            {
                let mem_sums = membership.mapv(|val: f64| val.powf(q)).sum();
                let fac = membership.mapv(|val| val.powf(q) / mem_sums);
                cluster.assign(&fac.dot(&data.mapv(|val| f64::from(val))).mapv(|val| T::from(val)));
            }
        }

        clusters
    }
}
