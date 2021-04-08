#![allow(non_snake_case)]

extern crate nalgebra as na;
use na::allocator::Allocator;
use na::dimension::Dim;
use na::{DefaultAllocator, MatrixN, VectorN};

pub struct KalmanFilter<D>
where
    D: Dim,
    DefaultAllocator: Allocator<f64, D, D>,
{
    /// システムの状態遷移モデル
    F: MatrixN<f64, D>,
    /// システムのノイズ
    Q: MatrixN<f64, D>,
    /// 観測のノイズ(
    R: MatrixN<f64, D>,
}

impl<D> KalmanFilter<D>
where
    D: Dim,
    DefaultAllocator: Allocator<f64, D, D> + Allocator<f64, D>,
{
    pub fn new(F: MatrixN<f64, D>, Q: MatrixN<f64, D>, R: MatrixN<f64, D>) -> Self {
        Self { F, Q, R }
    }

    /// 前の状態から現在の状態を推定する関数
    /// # Arguments
    /// - `x` : 以前の状態
    /// - `P` : 以前の誤差
    /// # Returns
    /// - `Vector4<f64>` - 推定された現在状態
    /// - `Matrix4<f64>` - 推定された現在誤差
    pub fn predict(
        &self,
        x: &VectorN<f64, D>,
        P: &MatrixN<f64, D>,
    ) -> (VectorN<f64, D>, MatrixN<f64, D>) {
        let x_pred = &self.F * x;
        let P_pred = &self.F * P * self.F.transpose() + &self.Q;
        (x_pred, P_pred)
    }

    /// 現在の観測から推定値を補正し状態を更新する関数
    /// # Arguments
    /// - `x` : 推定された現在状態
    /// - `P` : 推定された現在誤差
    /// - `z` : 観測された現在状態
    /// # Returns
    /// - `Vector4<f64>` : 更新された現在状態
    /// - `Matrix4<f64>` : 更新された現在誤差
    pub fn update(
        &self,
        x_pred: &VectorN<f64, D>,
        P_pred: &MatrixN<f64, D>,
        z: &VectorN<f64, D>,
    ) -> (VectorN<f64, D>, MatrixN<f64, D>) {
        // 観測残差
        let y = z - x_pred;
        // 観測残差の共分散
        let S = P_pred + &self.R;
        // 最適カルマンゲイン
        let K = P_pred * S.try_inverse().unwrap();

        // 値の更新
        let x_new = x_pred + &K * y;
        let P_new = &K * P_pred;

        (x_new, P_new)
    }
}
