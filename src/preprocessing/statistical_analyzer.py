"""
StatisticalAnalyzer - Detaylı istatistiksel analiz

Bu modül normallik testleri, stationarity testleri ve korelasyon
analizleri gerçekleştirir.
"""

import pandas as pd
import numpy as np
import logging
import warnings
from typing import Dict, Any, List, Optional
from scipy.stats import shapiro, anderson, kstest, normaltest
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from .config import STATISTICAL_CONFIG

warnings.filterwarnings("ignore")


class StatisticalAnalyzer:
    """
    Bitcoin fiyat verileri için detaylı istatistiksel analiz yapar.

    Attributes:
        data (pd.DataFrame): Analiz edilecek veri
        analysis_results (Dict): İstatistiksel analiz sonuçları
        logger (logging.Logger): Logging objesi
    """

    def __init__(self, log_level: str = "INFO"):
        """
        StatisticalAnalyzer'ı başlatır.

        Args:
            log_level (str): Logging seviyesi
        """
        self.data: Optional[pd.DataFrame] = None
        self.analysis_results: Dict[str, Any] = {}
        self.logger = self._setup_logger(log_level)

    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Logger'ı kurar."""
        logger = logging.getLogger(f"{__name__}.StatisticalAnalyzer")
        logger.setLevel(getattr(logging, log_level.upper()))

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def set_data(self, data: pd.DataFrame) -> None:
        """
        Analiz edilecek veriyi ayarlar.

        Args:
            data (pd.DataFrame): Analiz edilecek veri
        """
        self.data = data.copy()
        self.logger.info(f"İstatistiksel analiz için veri ayarlandı: {self.data.shape}")

    def test_normality(
        self, columns: Optional[List[str]] = None, alpha: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Normallik testleri gerçekleştirir.

        Args:
            columns (List[str], optional): Test edilecek sütunlar
            alpha (float, optional): Anlamlılık seviyesi

        Returns:
            Dict[str, Any]: Normallik test sonuçları
        """
        try:
            if alpha is None:
                alpha = float(STATISTICAL_CONFIG.get("normality_alpha", 0.05))  # type: ignore

            if columns is None:
                columns = ["open", "high", "low", "close", "volume"]

            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            self.logger.info(f"Normallik testleri başlatılıyor (alpha: {alpha})")

            results = {}

            for col in columns:
                if col in self.data.columns:
                    # Veriyi hazırla
                    data_clean = self.data[col].dropna()

                    if len(data_clean) < 3:
                        self.logger.warning(f"{col} sütunu için yeterli veri yok")
                        continue

                    col_results = {}

                    # Shapiro-Wilk testi
                    try:
                        shapiro_stat, shapiro_p = shapiro(data_clean)
                        col_results["shapiro_wilk"] = {
                            "statistic": shapiro_stat,
                            "p_value": shapiro_p,
                            "is_normal": shapiro_p > alpha,
                        }
                    except Exception as e:
                        self.logger.warning(
                            f"Shapiro-Wilk testi hatası ({col}): {str(e)}"
                        )
                        col_results["shapiro_wilk"] = {"error": str(e)}

                    # Anderson-Darling testi
                    try:
                        anderson_result = anderson(data_clean, dist="norm")
                        col_results["anderson_darling"] = {
                            "statistic": anderson_result.statistic,
                            "critical_values": anderson_result.critical_values.tolist(),
                            "significance_levels": anderson_result.significance_level.tolist(),
                            "is_normal": anderson_result.statistic
                            < anderson_result.critical_values[2],
                        }
                    except Exception as e:
                        self.logger.warning(
                            f"Anderson-Darling testi hatası ({col}): {str(e)}"
                        )
                        col_results["anderson_darling"] = {"error": str(e)}

                    # Kolmogorov-Smirnov testi
                    try:
                        ks_stat, ks_p = kstest(
                            data_clean,
                            "norm",
                            args=(data_clean.mean(), data_clean.std()),
                        )
                        col_results["kolmogorov_smirnov"] = {
                            "statistic": ks_stat,
                            "p_value": ks_p,
                            "is_normal": ks_p > alpha,
                        }
                    except Exception as e:
                        self.logger.warning(
                            f"Kolmogorov-Smirnov testi hatası ({col}): {str(e)}"
                        )
                        col_results["kolmogorov_smirnov"] = {"error": str(e)}

                    # D'Agostino testi
                    try:
                        dagostino_stat, dagostino_p = normaltest(data_clean)
                        col_results["dagostino"] = {
                            "statistic": dagostino_stat,
                            "p_value": dagostino_p,
                            "is_normal": dagostino_p > alpha,
                        }
                    except Exception as e:
                        self.logger.warning(
                            f"D'Agostino testi hatası ({col}): {str(e)}"
                        )
                        col_results["dagostino"] = {"error": str(e)}

                    # Genel normallik değerlendirmesi
                    normal_tests = [
                        k for k, v in col_results.items() if "is_normal" in v
                    ]
                    normal_count = sum(
                        1
                        for k in normal_tests
                        if col_results[k].get("is_normal", False)
                    )
                    col_results["overall_normal"] = {
                        "value": bool(normal_count >= len(normal_tests) / 2)
                    }

                    results[col] = col_results

            self.analysis_results["normality_tests"] = results

            # Genel özet
            total_columns = len(results)
            normal_columns = sum(
                1 for r in results.values() if r.get("overall_normal", False)
            )

            self.logger.info(
                f"Normallik testleri tamamlandı: {normal_columns}/{total_columns} sütun normal"
            )
            return results

        except Exception as e:
            self.logger.error(f"Normallik testleri hatası: {str(e)}")
            raise

    def test_stationarity(
        self, columns: Optional[List[str]] = None, alpha: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Stationarity testleri gerçekleştirir.

        Args:
            columns (List[str], optional): Test edilecek sütunlar
            alpha (float, optional): Anlamlılık seviyesi

        Returns:
            Dict[str, Any]: Stationarity test sonuçları
        """
        try:
            if alpha is None:
                alpha = float(STATISTICAL_CONFIG.get("stationarity_alpha", 0.05))  # type: ignore

            if columns is None:
                columns = ["open", "high", "low", "close", "volume"]

            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            self.logger.info(f"Stationarity testleri başlatılıyor (alpha: {alpha})")

            results = {}

            for col in columns:
                if col in self.data.columns:
                    # Veriyi hazırla
                    data_clean = self.data[col].dropna()

                    if len(data_clean) < 10:
                        self.logger.warning(f"{col} sütunu için yeterli veri yok")
                        continue

                    col_results = {}

                    # Augmented Dickey-Fuller testi
                    try:
                        adf_result = adfuller(
                            data_clean, maxlag=STATISTICAL_CONFIG["adf_max_lags"]
                        )
                        col_results["adf"] = {
                            "statistic": adf_result[0],
                            "p_value": adf_result[1],
                            "critical_values": adf_result[4],
                            "is_stationary": adf_result[1] < alpha,
                        }
                    except Exception as e:
                        self.logger.warning(f"ADF testi hatası ({col}): {str(e)}")
                        col_results["adf"] = {"error": str(e)}

                    # KPSS testi
                    try:
                        kpss_result = kpss(
                            data_clean, regression=STATISTICAL_CONFIG["kpss_regression"]
                        )
                        col_results["kpss"] = {
                            "statistic": kpss_result[0],
                            "p_value": kpss_result[1],
                            "critical_values": kpss_result[3],
                            "is_stationary": kpss_result[1] > alpha,
                        }
                    except Exception as e:
                        self.logger.warning(f"KPSS testi hatası ({col}): {str(e)}")
                        col_results["kpss"] = {"error": str(e)}

                    # Genel stationarity değerlendirmesi
                    stationary_tests = [
                        k for k, v in col_results.items() if "is_stationary" in v
                    ]
                    stationary_count = sum(
                        1
                        for k in stationary_tests
                        if col_results[k].get("is_stationary", False)
                    )
                    col_results["overall_stationary"] = {
                        "value": bool(stationary_count >= len(stationary_tests) / 2)
                    }

                    results[col] = col_results

            self.analysis_results["stationarity_tests"] = results

            # Genel özet
            total_columns = len(results)
            stationary_columns = sum(
                1 for r in results.values() if r.get("overall_stationary", False)
            )

            self.logger.info(
                f"Stationarity testleri tamamlandı: {stationary_columns}/{total_columns} sütun stationary"
            )
            return results

        except Exception as e:
            self.logger.error(f"Stationarity testleri hatası: {str(e)}")
            raise

    def analyze_correlations(
        self, columns: Optional[List[str]] = None, threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Korelasyon analizi gerçekleştirir.

        Args:
            columns (List[str], optional): Analiz edilecek sütunlar
            threshold (float, optional): Yüksek korelasyon threshold'u

        Returns:
            Dict[str, Any]: Korelasyon analiz sonuçları
        """
        try:
            if threshold is None:
                threshold = float(STATISTICAL_CONFIG.get("correlation_threshold", 0.7))  # type: ignore

            if columns is None:
                columns = ["open", "high", "low", "close", "volume"]

            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            self.logger.info(
                f"Korelasyon analizi başlatılıyor (threshold: {threshold})"
            )

            # Veriyi hazırla
            data_subset = self.data[columns].dropna()

            if len(data_subset) == 0:
                self.logger.warning("Analiz edilecek veri bulunamadı")
                return {}

            results = {}

            # Pearson korelasyonu
            try:
                pearson_corr = data_subset.corr(method="pearson")
                results["pearson"] = {
                    "correlation_matrix": pearson_corr.to_dict(),
                    "high_correlations": self._find_high_correlations(
                        pearson_corr, threshold
                    ),
                    "mean_correlation": pearson_corr.values[
                        np.triu_indices_from(pearson_corr.values, k=1)
                    ].mean(),
                }
            except Exception as e:
                self.logger.warning(f"Pearson korelasyon hatası: {str(e)}")
                results["pearson"] = {"error": str(e)}

            # Spearman korelasyonu
            try:
                spearman_corr = data_subset.corr(method="spearman")
                results["spearman"] = {
                    "correlation_matrix": spearman_corr.to_dict(),
                    "high_correlations": self._find_high_correlations(
                        spearman_corr, threshold
                    ),
                    "mean_correlation": spearman_corr.values[
                        np.triu_indices_from(spearman_corr.values, k=1)
                    ].mean(),
                }
            except Exception as e:
                self.logger.warning(f"Spearman korelasyon hatası: {str(e)}")
                results["spearman"] = {"error": str(e)}

            # Kendall korelasyonu
            try:
                kendall_corr = data_subset.corr(method="kendall")
                results["kendall"] = {
                    "correlation_matrix": kendall_corr.to_dict(),
                    "high_correlations": self._find_high_correlations(
                        kendall_corr, threshold
                    ),
                    "mean_correlation": kendall_corr.values[
                        np.triu_indices_from(kendall_corr.values, k=1)
                    ].mean(),
                }
            except Exception as e:
                self.logger.warning(f"Kendall korelasyon hatası: {str(e)}")
                results["kendall"] = {"error": str(e)}

            self.analysis_results["correlations"] = results

            # Genel özet
            high_corr_count = sum(
                len(r.get("high_correlations", []))
                for r in results.values()
                if "high_correlations" in r
            )

            self.logger.info(
                f"Korelasyon analizi tamamlandı: {high_corr_count} yüksek korelasyon"
            )
            return results

        except Exception as e:
            self.logger.error(f"Korelasyon analizi hatası: {str(e)}")
            raise

    def analyze_autocorrelation(
        self, columns: Optional[List[str]] = None, lags: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Autocorrelation analizi gerçekleştirir.

        Args:
            columns (List[str], optional): Analiz edilecek sütunlar
            lags (int, optional): Lag sayısı

        Returns:
            Dict[str, Any]: Autocorrelation analiz sonuçları
        """
        try:
            if lags is None:
                lags = int(STATISTICAL_CONFIG.get("autocorr_lags", 20))  # type: ignore

            if columns is None:
                columns = ["open", "high", "low", "close", "volume"]

            # Null safety kontrolü
            if self.data is None:
                return {"error": "Data is None"}

            self.logger.info(f"Autocorrelation analizi başlatılıyor (lags: {lags})")

            results = {}

            for col in columns:
                if col in self.data.columns:
                    # Veriyi hazırla
                    data_clean = self.data[col].dropna()

                    if len(data_clean) < (lags or 20) + 1:
                        self.logger.warning(f"{col} sütunu için yeterli veri yok")
                        continue

                    col_results = {}

                    # ACF analizi
                    try:
                        acf_values, acf_confint = acf(
                            data_clean, nlags=lags, alpha=0.05
                        )
                        col_results["acf"] = {
                            "values": acf_values.tolist(),
                            "confidence_intervals": acf_confint.tolist(),
                            "significant_lags": [
                                i
                                for i, (val, (lower, upper)) in enumerate(
                                    zip(acf_values, acf_confint)
                                )
                                if i > 0 and (val > upper or val < lower)
                            ],
                        }
                    except Exception as e:
                        self.logger.warning(f"ACF analizi hatası ({col}): {str(e)}")
                        col_results["acf"] = {"error": str(e)}

                    # PACF analizi
                    try:
                        pacf_values, pacf_confint = pacf(
                            data_clean, nlags=lags, alpha=0.05
                        )
                        col_results["pacf"] = {
                            "values": pacf_values.tolist(),
                            "confidence_intervals": pacf_confint.tolist(),
                            "significant_lags": [
                                i
                                for i, (val, (lower, upper)) in enumerate(
                                    zip(pacf_values, pacf_confint)
                                )
                                if i > 0 and (val > upper or val < lower)
                            ],
                        }
                    except Exception as e:
                        self.logger.warning(f"PACF analizi hatası ({col}): {str(e)}")
                        col_results["pacf"] = {"error": str(e)}

                    # Ljung-Box testi
                    try:
                        lb_result = acorr_ljungbox(
                            data_clean, lags=lags, return_df=True
                        )
                        col_results["ljung_box"] = {
                            "statistics": lb_result["lb_stat"].tolist(),
                            "p_values": lb_result["lb_pvalue"].tolist(),
                            "significant_lags": [
                                i
                                for i, p in enumerate(lb_result["lb_pvalue"])
                                if p < 0.05
                            ],
                        }
                    except Exception as e:
                        self.logger.warning(f"Ljung-Box testi hatası ({col}): {str(e)}")
                        col_results["ljung_box"] = {"error": str(e)}

                    results[col] = col_results

            self.analysis_results["autocorrelation"] = results

            # Genel özet
            total_columns = len(results)
            significant_columns = sum(
                1
                for r in results.values()
                if any(
                    "significant_lags" in v and len(v["significant_lags"]) > 0
                    for v in r.values()
                    if isinstance(v, dict)
                )
            )

            self.logger.info(
                f"Autocorrelation analizi tamamlandı: {significant_columns}/{total_columns} sütun anlamlı"
            )
            return results

        except Exception as e:
            self.logger.error(f"Autocorrelation analizi hatası: {str(e)}")
            raise

    def _find_high_correlations(
        self, corr_matrix: pd.DataFrame, threshold: float
    ) -> List[Dict]:
        """
        Yüksek korelasyonları bulur.

        Args:
            corr_matrix (pd.DataFrame): Korelasyon matrisi
            threshold (float): Threshold değeri

        Returns:
            List[Dict]: Yüksek korelasyonlar
        """
        high_correlations = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_correlations.append(
                        {
                            "variable1": corr_matrix.columns[i],
                            "variable2": corr_matrix.columns[j],
                            "correlation": corr_value,
                        }
                    )

        return high_correlations

    def run_all_analyses(self) -> Dict[str, Any]:
        """
        Tüm istatistiksel analizleri çalıştırır.

        Returns:
            Dict[str, Any]: Tüm analiz sonuçları
        """
        try:
            self.logger.info("Kapsamlı istatistiksel analiz başlatılıyor")

            # Tüm analizleri çalıştır
            self.test_normality()
            self.test_stationarity()
            self.analyze_correlations()
            self.analyze_autocorrelation()

            # Genel özet
            self.analysis_results["summary"] = {
                "total_analyses": len(self.analysis_results),
                "timestamp": pd.Timestamp.now().isoformat(),
                "data_quality": self._assess_data_quality(),
            }

            self.logger.info("İstatistiksel analiz tamamlandı")
            return self.analysis_results

        except Exception as e:
            self.logger.error(f"İstatistiksel analiz hatası: {str(e)}")
            raise

    def _assess_data_quality(self) -> str:
        """
        Veri kalitesini değerlendirir.

        Returns:
            str: Veri kalitesi değerlendirmesi
        """
        try:
            quality_score = 0

            # Normallik değerlendirmesi
            if "normality_tests" in self.analysis_results:
                normal_count = sum(
                    1
                    for r in self.analysis_results["normality_tests"].values()
                    if r.get("overall_normal", False)
                )
                total_count = len(self.analysis_results["normality_tests"])
                if total_count > 0:
                    quality_score += int((normal_count / total_count) * 25)

            # Stationarity değerlendirmesi
            if "stationarity_tests" in self.analysis_results:
                stationary_count = sum(
                    1
                    for r in self.analysis_results["stationarity_tests"].values()
                    if r.get("overall_stationary", False)
                )
                total_count = len(self.analysis_results["stationarity_tests"])
                if total_count > 0:
                    quality_score += int((stationary_count / total_count) * 25)

            # Korelasyon değerlendirmesi
            if "correlations" in self.analysis_results:
                high_corr_count = sum(
                    len(r.get("high_correlations", []))
                    for r in self.analysis_results["correlations"].values()
                    if "high_correlations" in r
                )
                if high_corr_count < 5:  # Az sayıda yüksek korelasyon iyi
                    quality_score += 25
                else:
                    quality_score += int(max(0, 25 - (high_corr_count - 5) * 2))

            # Autocorrelation değerlendirmesi
            if "autocorrelation" in self.analysis_results:
                significant_count = sum(
                    1
                    for r in self.analysis_results["autocorrelation"].values()
                    if any(
                        "significant_lags" in v and len(v["significant_lags"]) > 0
                        for v in r.values()
                        if isinstance(v, dict)
                    )
                )
                total_count = len(self.analysis_results["autocorrelation"])
                if total_count > 0:
                    quality_score += int((significant_count / total_count) * 25)

            # Kalite seviyesini belirle
            if quality_score >= 80:
                return "Excellent"
            elif quality_score >= 60:
                return "Good"
            elif quality_score >= 40:
                return "Fair"
            else:
                return "Poor"

        except Exception as e:
            self.logger.error(f"Veri kalitesi değerlendirme hatası: {str(e)}")
            return "Unknown"

    def get_analysis_report(self) -> Dict[str, Any]:
        """
        İstatistiksel analiz raporunu döndürür.

        Returns:
            Dict[str, Any]: Detaylı analiz raporu
        """
        return {
            "timestamp": pd.Timestamp.now().isoformat(),
            "data_shape": self.data.shape if self.data is not None else None,
            "analysis_results": self.analysis_results,
            "summary": {
                "total_analyses": len(self.analysis_results),
                "data_quality": self._assess_data_quality(),
                "recommendations": self._generate_recommendations(),
            },
        }

    def _generate_recommendations(self) -> List[str]:
        """
        Analiz sonuçlarına göre öneriler üretir.

        Returns:
            List[str]: Öneriler listesi
        """
        recommendations = []

        try:
            # Normallik önerileri
            if "normality_tests" in self.analysis_results:
                normal_count = sum(
                    1
                    for r in self.analysis_results["normality_tests"].values()
                    if r.get("overall_normal", False)
                )
                total_count = len(self.analysis_results["normality_tests"])
                if total_count > 0 and normal_count / total_count < 0.5:
                    recommendations.append(
                        "Veri normal dağılım göstermiyor, transformation gerekebilir"
                    )

            # Stationarity önerileri
            if "stationarity_tests" in self.analysis_results:
                stationary_count = sum(
                    1
                    for r in self.analysis_results["stationarity_tests"].values()
                    if r.get("overall_stationary", False)
                )
                total_count = len(self.analysis_results["stationarity_tests"])
                if total_count > 0 and stationary_count / total_count < 0.5:
                    recommendations.append(
                        "Veri stationary değil, differencing gerekebilir"
                    )

            # Korelasyon önerileri
            if "correlations" in self.analysis_results:
                high_corr_count = sum(
                    len(r.get("high_correlations", []))
                    for r in self.analysis_results["correlations"].values()
                    if "high_correlations" in r
                )
                if high_corr_count > 10:
                    recommendations.append(
                        "Yüksek korelasyon sayısı, multicollinearity kontrolü gerekebilir"
                    )

            # Autocorrelation önerileri
            if "autocorrelation" in self.analysis_results:
                significant_count = sum(
                    1
                    for r in self.analysis_results["autocorrelation"].values()
                    if any(
                        "significant_lags" in v and len(v["significant_lags"]) > 0
                        for v in r.values()
                        if isinstance(v, dict)
                    )
                )
                if significant_count > 0:
                    recommendations.append(
                        "Anlamlı autocorrelation tespit edildi, ARIMA modelleri uygun olabilir"
                    )

        except Exception as e:
            self.logger.error(f"Öneri üretme hatası: {str(e)}")
            recommendations.append("Analiz sonuçları değerlendirilemedi")

        return recommendations
