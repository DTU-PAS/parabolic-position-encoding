use crate::event::Event;
use crate::token::Token;

pub struct SpikingPatch {
    events: Vec<Event>,
    last_spike_time: Option<u64>,
    patch_x: u16,
    patch_y: u16,
    potential: usize,
    refractory_period: u64,
    threshold: usize,
}

impl SpikingPatch {
    pub fn new(patch_x: u16, patch_y: u16, refractory_period: u64, threshold: usize) -> Self {
        SpikingPatch {
            events: Vec::new(),
            last_spike_time: None,
            patch_x,
            patch_y,
            potential: 0,
            refractory_period,
            threshold,
        }
    }

    pub fn add(&mut self, event: Event) -> Option<Token> {
        let time = event.t;
        if self.in_refractory_period(time) {
            return None;
        }

        self.potential += 1;
        self.events.push(event);
        if self.potential >= self.threshold {
            return Some(self.spike(time));
        }
        None
    }

    pub fn reset(&mut self) {
        self.last_spike_time = None;
        self.potential = 0;
        self.events.clear();
    }

    fn in_refractory_period(&self, time: u64) -> bool {
        if let Some(last_spike_time) = self.last_spike_time {
            let difference = time - last_spike_time;
            return difference < self.refractory_period;
        }
        false
    }

    fn spike(&mut self, spike_time: u64) -> Token {
        self.last_spike_time = Some(spike_time);

        let x = self.patch_x;
        let y = self.patch_y;
        let t = spike_time;

        let events_x = self.events.iter().map(|e| e.x).collect();
        let events_y = self.events.iter().map(|e| e.y).collect();
        let events_t = self.events.iter().map(|e| e.t).collect();
        let events_p = self.events.iter().map(|e| e.p).collect();

        self.events.clear();
        self.potential = 0;

        (x, y, t, events_x, events_y, events_t, events_p)
    }
}
