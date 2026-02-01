mod event;
mod spiking_patch;
mod token;
mod tokenizer;

use token::{PyTokens, Tokens};

use numpy::ndarray::{ArrayBase, Dim, ViewRepr};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

type Location<'a> = ArrayBase<ViewRepr<&'a u16>, Dim<[usize; 1]>>;
type Time<'a> = ArrayBase<ViewRepr<&'a u64>, Dim<[usize; 1]>>;
type Polarity<'a> = ArrayBase<ViewRepr<&'a bool>, Dim<[usize; 1]>>;

#[pymodule]
fn spiking_patches(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBatchTokenizer>()?;
    m.add_class::<PyStreamingTokenizer>()?;
    Ok(())
}

#[pyclass(name = "BatchTokenizer")]
struct PyBatchTokenizer {
    height: usize,
    patch_size: usize,
    refractory_period: u64,
    threshold: usize,
    width: usize,
}

#[pymethods]
impl PyBatchTokenizer {
    #[new]
    fn new(
        height: usize,
        patch_size: usize,
        refractory_period: u64,
        threshold: usize,
        width: usize,
    ) -> Self {
        PyBatchTokenizer {
            height,
            refractory_period,
            patch_size,
            threshold,
            width,
        }
    }

    fn tokenize_batch<'py>(
        &self,
        py: Python<'py>,
        batch: Vec<(
            PyReadonlyArray1<u16>,
            PyReadonlyArray1<u16>,
            PyReadonlyArray1<u64>,
            PyReadonlyArray1<bool>,
        )>,
    ) -> Vec<PyTokens<'py>> {
        let batch: Vec<(Location, Location, Time, Polarity)> = batch
            .iter()
            .map(|(x, y, t, p)| {
                let x = x.as_array();
                let y = y.as_array();
                let t = t.as_array();
                let p = p.as_array();
                (x, y, t, p)
            })
            .collect();

        let batch: Vec<Tokens> = py.detach(|| {
            batch
                .into_par_iter()
                .map(|(x, y, t, p)| {
                    let mut tokenizer = tokenizer::Tokenizer::new(
                        self.height,
                        self.patch_size,
                        self.refractory_period,
                        self.threshold,
                        self.width,
                    );
                    tokenizer.tokenize(x, y, t, p)
                })
                .collect()
        });

        batch
            .into_iter()
            .map(|tokenizer_output| tokens_to_python(py, tokenizer_output))
            .collect()
    }
}

#[pyclass(name = "StreamingTokenizer")]
struct PyStreamingTokenizer {
    tokenizer: tokenizer::Tokenizer,
}

#[pymethods]
impl PyStreamingTokenizer {
    #[new]
    fn new(
        height: usize,
        patch_size: usize,
        refractory_period: u64,
        threshold: usize,
        width: usize,
    ) -> Self {
        let tokenizer =
            tokenizer::Tokenizer::new(height, patch_size, refractory_period, threshold, width);

        PyStreamingTokenizer { tokenizer }
    }

    fn reset(&mut self) {
        self.tokenizer.reset();
    }

    fn stream<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray1<u16>,
        y: PyReadonlyArray1<u16>,
        t: PyReadonlyArray1<u64>,
        p: PyReadonlyArray1<bool>,
    ) -> PyTokens<'py> {
        let x = x.as_array();
        let y = y.as_array();
        let t = t.as_array();
        let p = p.as_array();

        let tokens = self.tokenizer.tokenize(x, y, t, p);
        tokens_to_python(py, tokens)
    }
}

fn tokens_to_python<'py>(py: Python<'py>, tokens: Tokens) -> PyTokens<'py> {
    let x = tokens.x.into_pyarray(py);
    let y = tokens.y.into_pyarray(py);
    let t = tokens.t.into_pyarray(py);

    let events_x = tokens
        .events_x
        .into_iter()
        .map(|arr| arr.into_pyarray(py))
        .collect::<Vec<_>>();

    let events_y = tokens
        .events_y
        .into_iter()
        .map(|arr| arr.into_pyarray(py))
        .collect::<Vec<_>>();

    let events_t = tokens
        .events_t
        .into_iter()
        .map(|arr| arr.into_pyarray(py))
        .collect::<Vec<_>>();

    let events_p = tokens
        .events_p
        .into_iter()
        .map(|arr| arr.into_pyarray(py))
        .collect::<Vec<_>>();

    (x, y, t, events_x, events_y, events_t, events_p)
}
