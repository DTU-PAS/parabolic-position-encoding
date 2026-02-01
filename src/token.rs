use numpy::PyArray1;
use pyo3::Bound;

pub type Token = (u16, u16, u64, Vec<u16>, Vec<u16>, Vec<u64>, Vec<bool>);

pub struct Tokens {
    pub x: Vec<u16>,
    pub y: Vec<u16>,
    pub t: Vec<u64>,
    pub events_x: Vec<Vec<u16>>,
    pub events_y: Vec<Vec<u16>>,
    pub events_t: Vec<Vec<u64>>,
    pub events_p: Vec<Vec<bool>>,
}

pub type PyTokens<'py> = (
    Bound<'py, PyArray1<u16>>,       // x
    Bound<'py, PyArray1<u16>>,       // y
    Bound<'py, PyArray1<u64>>,       // t
    Vec<Bound<'py, PyArray1<u16>>>,  // events.x
    Vec<Bound<'py, PyArray1<u16>>>,  // events.y
    Vec<Bound<'py, PyArray1<u64>>>,  // events.t
    Vec<Bound<'py, PyArray1<bool>>>, // events.p
);
