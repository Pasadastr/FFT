#include <complex>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <stdexcept>

class FFT {
public:
    using Complex = std::complex<double>;
    using Vector = std::vector<Complex>;

    // Прямое ДПФ через байки Тыртыша
    static Vector forward(const Vector& x) {
        size_t N = x.size();
        if (N == 0) return {};
        
        // Цимес в k*n = (k^2 + n^2 - (k-n)^2) / 2, внезапно гугол говорит, что штука именная, некоего Блюстейна!
        // Тогда: X[k] = sum_n x[n] * exp(-2πi*k*n/N)
        // X[k] = exp(-πi*k^2/N) * sum_n (x[n] * exp(-πi*n^2/N)) * exp(πi*(k-n)^2/N)
        // a[n] = x[n]*exp(-πi*n^2/N) и b[n] = exp(πi*n^2/N)
        
        double angle_coeff = -M_PI / N;  // Знак для прямого ДПФ
        
        // Формируем последовательность a[n] = x[n] * exp(-πi*n^2/N)
        Vector a(N);
        for (size_t n = 0; n < N; ++n) {
            double angle = angle_coeff * n * n;
            a[n] = x[n] * Complex(std::cos(angle), std::sin(angle));
        }
        
        // Формируем последовательность b[n] = exp(πi*n^2/N)
        Vector b(2 * N - 1);
        for (size_t n = 0; n < 2 * N - 1; ++n) {
            // b[n] для n от -(N-1) до (N-1)
            int shifted_n = static_cast<int>(n) - static_cast<int>(N) + 1;
            double angle = -angle_coeff * shifted_n * shifted_n;  // exp(πi*(k-n)^2/N)
            b[n] = Complex(std::cos(angle), std::sin(angle));
        }
        
        // Вычисляем свёртку a * b через вложение тёплицевой матрицы в циркулянтную
        Vector conv = convolve_circulant(a, b);
        
        // Извлекаем результат: X[k] = exp(-πi*k^2/N) * conv[k + N - 1]
        Vector result(N);
        for (size_t k = 0; k < N; ++k) {
            double angle = angle_coeff * k * k;
            Complex chirp(std::cos(angle), std::sin(angle));
            result[k] = chirp * conv[k + N - 1];
        }
        
        return result;
    }

    // Обратное ДПФ 
    static Vector inverse(const Vector& x) {
        size_t N = x.size();
        if (N == 0) return {};
        
        // Для обратного ДПФ меняем знак в экспоненте и добавляем нормализацию
        double angle_coeff = M_PI / N;  // Знак для обратного ДПФ
        
        // a[n] = x[n] * exp(πi*n^2/N)
        Vector a(N);
        for (size_t n = 0; n < N; ++n) {
            double angle = angle_coeff * n * n;
            a[n] = x[n] * Complex(std::cos(angle), std::sin(angle));
        }
        
        // b[n] = exp(-πi*n^2/N)
        Vector b(2 * N - 1);
        for (size_t n = 0; n < 2 * N - 1; ++n) {
            int shifted_n = static_cast<int>(n) - static_cast<int>(N) + 1;
            double angle = -angle_coeff * shifted_n * shifted_n;
            b[n] = Complex(std::cos(angle), std::sin(angle));
        }
        
        // Свёртка
        Vector conv = convolve_circulant(a, b);
        
        // X[k] = (1/N) * exp(πi*k^2/N) * conv[k + N - 1]
        Vector result(N);
        double inv_N = 1.0 / N;
        for (size_t k = 0; k < N; ++k) {
            double angle = angle_coeff * k * k;
            Complex chirp(std::cos(angle), std::sin(angle));
            result[k] = chirp * conv[k + N - 1] * inv_N;
        }
        
        return result;
    }

private:
    // Вычисление свёртки через вложение тёплицевой матрицы в циркулянтную
    static Vector convolve_circulant(const Vector& a, const Vector& b) {
        size_t N = a.size();
        size_t M = b.size();  // M = 2N - 1
        
        // Размер циркулянтной матрицы: выбираем степень двойки >= 2*N-1
        size_t L = 1;
        while (L < 2 * N - 1) L <<= 1;
        
        // Дополняем нулями до размера L
        Vector a_padded(L, Complex(0, 0));
        Vector b_padded(L, Complex(0, 0));
        
        std::copy(a.begin(), a.end(), a_padded.begin());
        std::copy(b.begin(), b.end(), b_padded.begin());
        
        // Вычисляем БПФ обеих последовательностей
        Vector A = fft(a_padded);
        Vector B = fft(b_padded);
        
        // Поэлементное умножение в частотной области
        for (size_t i = 0; i < L; ++i) {
            A[i] *= B[i];
        }
        
        // Обратное БПФ даёт циклическую свёртку
        Vector cyclic_conv = ifft(A);
        
        // Из-за zero-padding циклическая свёртка равна линейной
        Vector result(2 * N - 1);
        for (size_t i = 0; i < 2 * N - 1; ++i) {
            result[i] = cyclic_conv[i];
        }
        
        return result;
    }

    // СОснова для степеней двоечки
    static Vector fft(const Vector& a) {
        size_t n = a.size();
        if (n <= 1) return a;
        
        if ((n & (n - 1)) != 0) {
            throw std::runtime_error("FFT size must be power of 2");
        }
        
        Vector a_even(n / 2), a_odd(n / 2);
        for (size_t i = 0; i < n / 2; ++i) {
            a_even[i] = a[2 * i];
            a_odd[i] = a[2 * i + 1];
        }
        
        Vector y_even = fft(a_even);
        Vector y_odd = fft(a_odd);
        
        Vector y(n);
        for (size_t k = 0; k < n / 2; ++k) {
            double angle = -2.0 * M_PI * k / n;
            Complex w(std::cos(angle), std::sin(angle));
            
            y[k] = y_even[k] + w * y_odd[k];
            y[k + n / 2] = y_even[k] - w * y_odd[k];
        }
        
        return y;
    }

    // Обратное БПФ
    static Vector ifft(const Vector& a) {
        size_t n = a.size();
        
        Vector conj_a(n);
        for (size_t i = 0; i < n; ++i) {
            conj_a[i] = std::conj(a[i]);
        }
        
        Vector y = fft(conj_a);
        
        for (size_t i = 0; i < n; ++i) {
            y[i] = std::conj(y[i]) / static_cast<double>(n);
        }
        
        return y;
    }
};

// Для проверки на вшивость:
// Прямое умножение на матрицу Фурье
FFT::Vector direct_dft(const FFT::Vector& x, bool inverse) {
    size_t N = x.size();
    FFT::Vector y(N, {0, 0});
    double sign = inverse ? 1.0 : -1.0;
    double norm = inverse ? 1.0 / N : 1.0;
    
    for (size_t k = 0; k < N; ++k) {
        for (size_t n = 0; n < N; ++n) {
            double angle = sign * 2.0 * M_PI * k * n / N;
            FFT::Complex w(std::cos(angle), std::sin(angle));
            y[k] += x[n] * w;
        }
        y[k] *= norm;
    }
    return y;
}

FFT::Vector generate_random(size_t n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    FFT::Vector data(n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = FFT::Complex(dis(gen), dis(gen));
    }
    return data;
}

double mse(const FFT::Vector& a, const FFT::Vector& b) {
    double mse = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        auto diff = a[i] - b[i];
        mse += std::norm(diff);
    }
    return mse / a.size();
}



int main() {
    std::cout << std::fixed << std::setprecision(15);
    
/*    
    // Глянем с норм умножением
    std::cout << "--- Test 1: Forward FFT vs Direct DFT ---\n";
    std::vector<size_t> test_sizes = {2, 3, 5, 6, 7, 10, 16, 17, 32, 60, 100};
    
    for (size_t N : test_sizes) {
        auto x = generate_random(N);
        
        auto y_tee = FFT::forward(x);
        auto y_direct = direct_dft(x, false);
        
        double mse = mse(y_tee, y_direct);

        std::cout << "N = " << std::setw(3) << N 
                  << "  MSE: " << std::scientific << mse;
    }
    
*/

    std::cout << "\n--- Смотр ошибки меж изначальным вектором и опосля ifft(fft()) ---\n";
    
    std::vector<size_t> test_sizes = {2, 3, 5, 6, 7, 10, 16, 17, 32, 60, 100};    
    
    
    for (size_t N : test_sizes) {
        auto original = generate_random(N);
        
        auto freq = FFT::forward(original);
        auto reconstructed = FFT::inverse(freq);
        
        double MSE = mse(original, reconstructed);

        std::cout << "Длина вектора = " << std::setw(3) << N 
                  << "  MSE: " << std::scientific << MSE << "\n";
    }
    
    
    return 0;
}
