use rand::prelude::*;

use std;
use super::*;
use search_tree::*;

#[derive(Clone)]
pub struct PolicyRng {
    pub rng: SmallRng
}

impl PolicyRng {
    pub fn new() -> Self {
        let rng = SeedableRng::seed_from_u64(0xF00DBABEC0FFEE60);
        Self {rng}
    }

    pub fn select_by_key<T, Iter, KeyFn>(&mut self, elts: Iter, mut key_fn: KeyFn) -> Option<T>
        where Iter: Iterator<Item=T>, KeyFn: FnMut(&T) -> f32
    {
        let mut choice = None;
        let mut num_optimal = 0;
        let mut best_so_far: f32 = std::f32::NEG_INFINITY;
        for elt in elts {
            let score = key_fn(&elt);
            if score > best_so_far {
                choice = Some(elt);
                num_optimal = 1;
                best_so_far = score;
            } else if score == best_so_far {
                num_optimal += 1;
                if self.rng.gen_ratio(1.min(num_optimal), num_optimal) {
                    choice = Some(elt);
                }
            }
        }
        choice
    }
}

impl Default for PolicyRng {
    fn default() -> Self {
        Self::new()
    }
}

pub trait TreePolicy<Spec: MCTS<TreePolicy=Self>>: Sync + Sized {
    type MoveEvaluation: Sync + Send;
    type ThreadLocalData: Default;

    fn choose_child<'a, MoveIter>(&self, moves: MoveIter, handle: SearchHandle<Spec>) -> &'a MoveInfo<Spec>
        where MoveIter: Iterator<Item=&'a MoveInfo<Spec>> + Clone;
    fn validate_evaluations(&self, _evalns: &[Self::MoveEvaluation]) {}
}

#[derive(Clone, Debug)]
pub struct UCTPolicy {
    exploration_constant: f32,
}

impl UCTPolicy {
    pub fn new(exploration_constant: f32) -> Self {
        assert!(exploration_constant > 0.0,
            "exploration constant is {} (must be positive)",
            exploration_constant);
        Self {exploration_constant}
    }

    pub fn exploration_constant(&self) -> f32 {
        self.exploration_constant
    }
}

impl<Spec: MCTS<TreePolicy=Self>> TreePolicy<Spec> for UCTPolicy
{
    type ThreadLocalData = PolicyRng;
    type MoveEvaluation = ();

    fn choose_child<'a, MoveIter>(&self, moves: MoveIter, mut handle: SearchHandle<Spec>) -> &'a MoveInfo<Spec>
        where MoveIter: Iterator<Item=&'a MoveInfo<Spec>> + Clone
    {
        let total_visits = moves.clone().map(|x| x.visits()).sum::<u64>();
        let adjusted_total = (total_visits + 1) as f32;
        let ln_adjusted_total = adjusted_total.ln();
        handle.thread_data().policy_data.select_by_key(moves, |mov| {
            let sum_rewards = mov.sum_rewards();
            let child_visits = mov.visits();
            // http://mcts.ai/pubs/mcts-survey-master.pdf
            if child_visits == 0 {
                std::f32::INFINITY
            } else {
                let explore_term = 2.0 * (ln_adjusted_total / child_visits as f32).sqrt();
                let mean_action_value = sum_rewards as f32 / child_visits as f32;
                self.exploration_constant * explore_term + mean_action_value
            }
        }).unwrap()
    }
}

const RECIPROCAL_TABLE_LEN: usize = 128;

#[derive(Clone, Debug)]
pub struct AlphaGoPolicy {
    exploration_constant: f32,
    reciprocals: Vec<f32>,
}

impl AlphaGoPolicy {
    pub fn new(exploration_constant: f32) -> Self {
        assert!(exploration_constant > 0.0,
            "exploration constant is {} (must be positive)",
            exploration_constant);
        let reciprocals = (0..RECIPROCAL_TABLE_LEN)
            .map(|x| if x == 0 {
                2.0
            } else {
                1.0 / x as f32
            })
            .collect();
        Self {exploration_constant, reciprocals}
    }

    pub fn exploration_constant(&self) -> f32 {
        self.exploration_constant
    }

    fn reciprocal(&self, x: usize) -> f32 {
        if x < RECIPROCAL_TABLE_LEN {
            unsafe {
                *self.reciprocals.get_unchecked(x)
            }
        } else {
            1.0 / x as f32
        }
    }
}

impl<Spec: MCTS<TreePolicy=Self>> TreePolicy<Spec> for AlphaGoPolicy
{
    type ThreadLocalData = PolicyRng;
    type MoveEvaluation = f32;

    fn choose_child<'a, MoveIter>(&self, moves: MoveIter, mut handle: SearchHandle<Spec>) -> &'a MoveInfo<Spec>
        where MoveIter: Iterator<Item=&'a MoveInfo<Spec>> + Clone
    {
        let total_visits = moves.clone().map(|x| x.visits()).sum::<u64>() + 1;
        let sqrt_total_visits = (total_visits as f32).sqrt();
        let explore_coef = self.exploration_constant * sqrt_total_visits;
        handle.thread_data().policy_data.select_by_key(moves, |mov| {
            let sum_rewards = mov.sum_rewards() as f32;
            let child_visits = mov.visits();
            let policy_evaln = *mov.move_evaluation() as f32;
            (sum_rewards + explore_coef * policy_evaln) * self.reciprocal(child_visits as usize)
        }).unwrap()
    }

    fn validate_evaluations(&self, evalns: &[f32]) {
        for &x in evalns {
            assert!(x >= -1e-6,
                "Move evaluation is {} (must be non-negative)",
                x);
        }
        if evalns.len() >= 1 {
            let evaln_sum: f32 = evalns.iter().sum();
            assert!((evaln_sum - 1.0).abs() < 0.1,
                "Sum of evaluations is {} (should sum to 1)",
                evaln_sum);
        }
    }
}
