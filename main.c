#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

// Utility
uint32_t GETCPUTIME()
{
    // Works up to 100 Million cycles
    uint32_t count;
    do {
        asm volatile ("csrr %0, mcycle\t" :  "=r"(count):);
    } while (count > (4194967295));

    return count;
}

// ---------------------------------------------------------
// Fixed-point configuration. We are using S7.8 format, refered to as S7_8 below. 
// ---------------------------------------------------------
#define Q_SHIFT 8               // Number of fractional bits
#define Q_SCALE (1 << Q_SHIFT)  // = 256
#define TO_Q(x) ((int16_t)((x) * Q_SCALE + ((x) >= 0 ? 0.5f : -0.5f))) // Helper macro to convert float -> Qm.n integer, for demonstration

// Hyperparams
#define SEQ_LEN   3
#define MODEL_DIM 4
#define FF_DIM    8

//mha hyperparams
#define HEADS  2
#define HEAD_DIM (MODEL_DIM/HEADS)



// Transformer Parameters
// Hard-coded projection weights (S7_8)
// Usually these come from training, but here we’re just making up numbers.

int16_t WQ[MODEL_DIM * MODEL_DIM] = {
    TO_Q(0.1f), TO_Q(0.2f), TO_Q(0.3f), TO_Q(0.4f),
    TO_Q(0.0f), TO_Q(0.1f), TO_Q(0.2f), TO_Q(0.3f),
    TO_Q(0.2f), TO_Q(0.0f), TO_Q(0.1f), TO_Q(0.3f),
    TO_Q(0.1f), TO_Q(0.3f), TO_Q(0.0f), TO_Q(0.2f)
};
int16_t WK[MODEL_DIM * MODEL_DIM] = {
    TO_Q(0.4f), TO_Q(0.3f), TO_Q(0.2f), TO_Q(0.1f),
    TO_Q(0.2f), TO_Q(0.2f), TO_Q(0.2f), TO_Q(0.2f),
    TO_Q(0.1f), TO_Q(0.0f), TO_Q(0.3f), TO_Q(0.2f),
    TO_Q(0.0f), TO_Q(0.1f), TO_Q(0.1f), TO_Q(0.4f)
};
int16_t WV[MODEL_DIM * MODEL_DIM] = {
    TO_Q(0.3f), TO_Q(0.1f), TO_Q(0.2f), TO_Q(0.0f),
    TO_Q(0.2f), TO_Q(0.2f), TO_Q(0.3f), TO_Q(0.1f),
    TO_Q(0.1f), TO_Q(0.3f), TO_Q(0.3f), TO_Q(0.2f),
    TO_Q(0.0f), TO_Q(0.2f), TO_Q(0.1f), TO_Q(0.2f)
};
/*
// Add 2 heads for wq, wk and wv
int16_t WQ[HEADS][HEAD_DIM * MODEL_DIM] = {
    // Head 0 weights
    {
        TO_Q(0.1f), TO_Q(0.2f), TO_Q(0.3f), TO_Q(0.4f),
        TO_Q(0.0f), TO_Q(0.1f), TO_Q(0.2f), TO_Q(0.3f)
    },
    // Head 1 weights
    {
        TO_Q(0.2f), TO_Q(0.0f), TO_Q(0.1f), TO_Q(0.3f),
        TO_Q(0.1f), TO_Q(0.3f), TO_Q(0.0f), TO_Q(0.2f)
    }
};
int16_t WK[HEADS][HEAD_DIM * MODEL_DIM] = {
    // Head 0 weights
    {
        TO_Q(0.4f), TO_Q(0.3f), TO_Q(0.2f), TO_Q(0.1f),
        TO_Q(0.2f), TO_Q(0.2f), TO_Q(0.2f), TO_Q(0.2f)
    },
    // Head 1 weights
    {
        TO_Q(0.1f), TO_Q(0.0f), TO_Q(0.3f), TO_Q(0.2f),
        TO_Q(0.0f), TO_Q(0.1f), TO_Q(0.1f), TO_Q(0.4f)
    }
};

int16_t WV[HEADS][HEAD_DIM * MODEL_DIM] = {
    // Head 0 weights
    {
        TO_Q(0.3f), TO_Q(0.1f), TO_Q(0.2f), TO_Q(0.0f),
        TO_Q(0.2f), TO_Q(0.2f), TO_Q(0.3f), TO_Q(0.1f)
    },
    // Head 1 weights
    {
        TO_Q(0.1f), TO_Q(0.3f), TO_Q(0.3f), TO_Q(0.2f),
        TO_Q(0.0f), TO_Q(0.2f), TO_Q(0.1f), TO_Q(0.2f)
    }
};
*/
// Feed-forward weights (S7_8)
int16_t W1[MODEL_DIM * FF_DIM] = {
    // shape = (FF_DIM=8, MODEL_DIM=4)
    TO_Q(0.1f), TO_Q(0.2f), TO_Q(0.0f), TO_Q(0.0f),
    TO_Q(0.3f), TO_Q(0.1f), TO_Q(0.1f), TO_Q(0.1f),
    TO_Q(0.2f), TO_Q(0.2f), TO_Q(0.4f), TO_Q(0.0f),
    TO_Q(0.0f), TO_Q(0.1f), TO_Q(0.1f), TO_Q(0.2f),
    TO_Q(0.1f), TO_Q(0.3f), TO_Q(0.0f), TO_Q(0.0f),
    TO_Q(0.1f), TO_Q(0.0f), TO_Q(0.1f), TO_Q(0.3f),
    TO_Q(0.0f), TO_Q(0.2f), TO_Q(0.2f), TO_Q(0.1f),
    TO_Q(0.2f), TO_Q(0.2f), TO_Q(0.0f), TO_Q(0.1f)
};
int16_t b1[FF_DIM] = {
    TO_Q(0.0f), TO_Q(0.1f), TO_Q(0.1f), TO_Q(0.0f),
    TO_Q(0.2f), TO_Q(0.0f), TO_Q(0.0f), TO_Q(0.1f)
};

int16_t W2[FF_DIM * MODEL_DIM] = {
    // shape = (MODEL_DIM=4, FF_DIM=8)
    TO_Q(0.1f), TO_Q(0.0f), TO_Q(0.2f), TO_Q(0.1f), 
    TO_Q(0.3f), TO_Q(0.2f), TO_Q(0.0f), TO_Q(0.0f),
    TO_Q(0.0f), TO_Q(0.1f), TO_Q(0.1f), TO_Q(0.3f), 
    TO_Q(0.2f), TO_Q(0.0f), TO_Q(0.2f), TO_Q(0.1f),
    TO_Q(0.0f), TO_Q(0.2f), TO_Q(0.3f), TO_Q(0.1f), 
    TO_Q(0.0f), TO_Q(0.1f), TO_Q(0.1f), TO_Q(0.1f),
    TO_Q(0.2f), TO_Q(0.1f), TO_Q(0.2f), TO_Q(0.0f), 
    TO_Q(0.2f), TO_Q(0.1f), TO_Q(0.3f), TO_Q(0.0f)
};
int16_t b2[MODEL_DIM] = {
    TO_Q(0.1f), TO_Q(0.1f), TO_Q(0.0f), TO_Q(0.2f)
};




//=======================================================================================================
//=======================================================================================================
//=======================================================================================================
//=======================================================================================================
//=======================================================================================================
// START OF HACKATHON CODE
//=======================================================================================================

// Saturate 32-bit intermediate to 16-bit range
static inline int16_t saturate_i16(int32_t x) {
    if (x >  32767) return  32767;
    if (x < -32768) return -32768;
    return (int16_t)x;
}

// ---------------------------------------------------------
// 1) Dot product (S7_8) => also produce a S7_8 result
// ---------------------------------------------------------
int16_t dot_S7_8( int16_t *a, int16_t *b, int size) {
    int32_t acc = 0;
    for(int i = 0; i < size; i++){
        // S7_8 x S7_8 => Q16; shift down => S7_8
        int32_t mul = (int32_t)a[i] * (int32_t)b[i];
        acc += (mul >> Q_SHIFT);
    }
    return saturate_i16(acc);
}


// ---------------------------------------------------------
// 2) Integer matrix-vector multiply: out = mat(rows x cols) * vec(cols x 1)
//    Both stored in S7_8; final result also in S7_8
// ---------------------------------------------------------
/*
void matvec_mul_S7_8(int16_t *mat, // [rows * cols] in S7_8
                   volatile int16_t *vec, // [cols] in S7_8
                   int16_t       *out, // [rows] in S7_8
                   int            rows,
                   int            cols)
{
    for (int r = 0; r < rows; r++) {
        // Accumulate in 32-bit
        int32_t acc = 0;
        for (int c = 0; c < cols; c++) {
            // S7_8 x S7_8 => Q16, then shift back to S7_8
            int32_t mul = (int32_t)mat[r * cols + c] * (int32_t)vec[c];
            // shift down by Q_SHIFT to return to S7_8
            acc += (mul >> Q_SHIFT); // Truncating
        }
        // saturate to int16
        out[r] = saturate_i16(acc);
    }
}
*/
// Optimizes variations include:
// loop unroll & better pipelining for fetch + decode
//29945 cycles

void matvec_mul_S7_8(int16_t *mat, 
                   volatile int16_t *vec, 
                   int16_t *out, 
                   int rows,
                   int cols)
{
    
    const int32_t shift = Q_SHIFT;
    const int UNROLL_FACTOR = 16;  
    const int32_t sat_min = -32768;
    const int32_t sat_max = 32767;
    //pragma omp parallel for  -> 16 unroll
    for (int r = 0; r < rows; r++) {
        int32_t acc = 0;
        int16_t *row_ptr = &mat[r * cols];
        int c;
        for (c = 0; c < cols - (UNROLL_FACTOR - 1); c += UNROLL_FACTOR) {
            int32_t sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
            
            // SIMD-Group 1
            sum1 += (int32_t)row_ptr[c] * (int32_t)vec[c];
            sum1 += (int32_t)row_ptr[c + 1] * (int32_t)vec[c + 1];
            sum1 += (int32_t)row_ptr[c + 2] * (int32_t)vec[c + 2];
            sum1 += (int32_t)row_ptr[c + 3] * (int32_t)vec[c + 3];
            
            // SIMD-Group 2
            sum2 += (int32_t)row_ptr[c + 4] * (int32_t)vec[c + 4];
            sum2 += (int32_t)row_ptr[c + 5] * (int32_t)vec[c + 5];
            sum2 += (int32_t)row_ptr[c + 6] * (int32_t)vec[c + 6];
            sum2 += (int32_t)row_ptr[c + 7] * (int32_t)vec[c + 7];
            
            // SIMD-Group 3
            sum3 += (int32_t)row_ptr[c + 8] * (int32_t)vec[c + 8];
            sum3 += (int32_t)row_ptr[c + 9] * (int32_t)vec[c + 9];
            sum3 += (int32_t)row_ptr[c + 10] * (int32_t)vec[c + 10];
            sum3 += (int32_t)row_ptr[c + 11] * (int32_t)vec[c + 11];
            
            // SIMD-Group 4
            sum4 += (int32_t)row_ptr[c + 12] * (int32_t)vec[c + 12];
            sum4 += (int32_t)row_ptr[c + 13] * (int32_t)vec[c + 13];
            sum4 += (int32_t)row_ptr[c + 14] * (int32_t)vec[c + 14];
            sum4 += (int32_t)row_ptr[c + 15] * (int32_t)vec[c + 15];
            
            acc += ((sum1 + sum2 + sum3 + sum4) >> shift);
        }
        for (; c < cols; c++) {
            acc += ((int32_t)row_ptr[c] * (int32_t)vec[c]) >> shift;
        }
        out[r] = (int16_t)((acc < sat_min) ? sat_min : (acc > sat_max) ? sat_max : acc);
    }
}

// 29320
/*
void matvec_mul_S7_8(int16_t *mat, 
                   volatile int16_t *vec, 
                   int16_t *out, 
                   int rows,
                   int cols)
{
    // Unroll the inner loop for better performance
    const int UNROLL_FACTOR = 4;
    int c;
    
    for (int r = 0; r < rows; r++) {
        int32_t acc = 0;
        
        // Process multiple columns at once
        for (c = 0; c < cols - (UNROLL_FACTOR - 1); c += UNROLL_FACTOR) {
            // Load 4 values at once
            int32_t mul0 = (int32_t)mat[r * cols + c] * (int32_t)vec[c];
            int32_t mul1 = (int32_t)mat[r * cols + c + 1] * (int32_t)vec[c + 1];
            int32_t mul2 = (int32_t)mat[r * cols + c + 2] * (int32_t)vec[c + 2];
            int32_t mul3 = (int32_t)mat[r * cols + c + 3] * (int32_t)vec[c + 3];
            
            // Accumulate with shift
            acc += ((mul0 + mul1 + mul2 + mul3) >> Q_SHIFT);
        }
        
        // Handle remaining columns
        for (; c < cols; c++) {
            int32_t mul = (int32_t)mat[r * cols + c] * (int32_t)vec[c];
            acc += (mul >> Q_SHIFT);
        }
        
        out[r] = saturate_i16(acc);
    }
}*/

// ---------------------------------------------------------
// 3) "Fake" Softmax in integer domain
//    - Actually: ReLU the scores, then normalize them so sum=Q_SCALE
//    - This is NOT a real exponent-based softmax, just a quick hack
// ---------------------------------------------------------
//29360 cycles
/*
void fake_softmax_S7_8(int16_t *values, int length) {
    // 3a) ReLU
    for(int i = 0; i < length; i++){
        if(values[i] < 0) values[i] = 0;
    }
    // 3b) Sum
    int32_t sum = 0;
    for(int i = 0; i < length; i++){
        sum += values[i];
    }
    // Avoid division by zero
    if(sum == 0) {
        // If all are zero, just set them uniform as Q_SCALE / length
        int16_t uniform = (int16_t)(Q_SCALE / length);
        for(int i = 0; i < length; i++){
            values[i] = uniform;
        }
        return;
    }
    // 3c) Normalize so that sum(weights)=Q_SCALE
    for(int i = 0; i < length; i++){
        // scale to Q_SCALE
        int32_t scaled = ((int32_t)values[i] * Q_SCALE) / sum;
        values[i] = saturate_i16(scaled);
    }
}
*/
// 29132
void fake_softmax_S7_8(int16_t *values, int length) {
    // Constants for optimization
    const int32_t q_scale = Q_SCALE;
    const int16_t zero = 0;
    const int CHUNK_SIZE = 4;
    
    // First pass: ReLU and sum calculation combined
    int32_t sum = 0;
    int i;
    
    // Pragma parallel -> 4 unroll
    for (i = 0; i < length - (CHUNK_SIZE - 1); i += CHUNK_SIZE) {
        // Process 4 elements at once
        int16_t v0 = values[i];
        int16_t v1 = values[i + 1];
        int16_t v2 = values[i + 2];
        int16_t v3 = values[i + 3];
        
        v0 = (v0 < zero) ? zero : v0;
        v1 = (v1 < zero) ? zero : v1;
        v2 = (v2 < zero) ? zero : v2;
        v3 = (v3 < zero) ? zero : v3;
        values[i] = v0;
        values[i + 1] = v1;
        values[i + 2] = v2;
        values[i + 3] = v3;
        sum += v0 + v1 + v2 + v3;
    }
    for (; i < length; i++) {
        int16_t v = values[i];
        v = (v < zero) ? zero : v;
        values[i] = v;
        sum += v;
    }
    
    if (sum == 0) {
        int16_t uniform = (int16_t)(q_scale / length);
        for (i = 0; i < length - (CHUNK_SIZE - 1); i += CHUNK_SIZE) {
            values[i] = uniform;
            values[i + 1] = uniform;
            values[i + 2] = uniform;
            values[i + 3] = uniform;
        }
        for (; i < length; i++) {
            values[i] = uniform;
        }
        return;
    }
    for (i = 0; i < length - (CHUNK_SIZE - 1); i += CHUNK_SIZE) {
        int32_t scaled0 = ((int32_t)values[i] * q_scale) / sum;
        int32_t scaled1 = ((int32_t)values[i + 1] * q_scale) / sum;
        int32_t scaled2 = ((int32_t)values[i + 2] * q_scale) / sum;
        int32_t scaled3 = ((int32_t)values[i + 3] * q_scale) / sum;
        
        values[i] = (int16_t)((scaled0 < -32768) ? -32768 : 
                            (scaled0 > 32767) ? 32767 : scaled0);
        values[i + 1] = (int16_t)((scaled1 < -32768) ? -32768 : 
                                (scaled1 > 32767) ? 32767 : scaled1);
        values[i + 2] = (int16_t)((scaled2 < -32768) ? -32768 : 
                                (scaled2 > 32767) ? 32767 : scaled2);
        values[i + 3] = (int16_t)((scaled3 < -32768) ? -32768 : 
                                (scaled3 > 32767) ? 32767 : scaled3);
    }
    for (; i < length; i++) {
        int32_t scaled = ((int32_t)values[i] * q_scale) / sum;
        values[i] = (int16_t)((scaled < -32768) ? -32768 : 
                            (scaled > 32767) ? 32767 : scaled);
    }
}
/* 29360
void fake_softmax_S7_8(int16_t *values, int length) {
    // Constants for optimization
    const int32_t q_scale = Q_SCALE;
    const int16_t zero = 0;
    const int CHUNK_SIZE = 8;
    int32_t sum = 0;
    
    // First pass: ReLU and sum calculation combined
    int i;
    // pragma omp parallel for -> 8 unroll
    for (i = 0; i < length - (CHUNK_SIZE - 1); i += CHUNK_SIZE) {
        int16_t v0 = values[i];
        int16_t v1 = values[i + 1];
        int16_t v2 = values[i + 2];
        int16_t v3 = values[i + 3];
        int16_t v4 = values[i + 4];
        int16_t v5 = values[i + 5];
        int16_t v6 = values[i + 6];
        int16_t v7 = values[i + 7];
        v0 = (v0 < zero) ? zero : v0;
        v1 = (v1 < zero) ? zero : v1;
        v2 = (v2 < zero) ? zero : v2;
        v3 = (v3 < zero) ? zero : v3;
        v4 = (v4 < zero) ? zero : v4;
        v5 = (v5 < zero) ? zero : v5;
        v6 = (v6 < zero) ? zero : v6;
        v7 = (v7 < zero) ? zero : v7;
        values[i] = v0;
        values[i + 1] = v1;
        values[i + 2] = v2;
        values[i + 3] = v3;
        values[i + 4] = v4;
        values[i + 5] = v5;
        values[i + 6] = v6;
        values[i + 7] = v7;
        sum += v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
    }
    for (; i < length; i++) {
        int16_t v = values[i];
        v = (v < zero) ? zero : v;
        values[i] = v;
        sum += v;
    }
    if (sum == 0) {
        int16_t uniform = (int16_t)(q_scale / length);
        // Process in chunks
        for (i = 0; i < length - (CHUNK_SIZE - 1); i += CHUNK_SIZE) {
            values[i] = uniform;
            values[i + 1] = uniform;
            values[i + 2] = uniform;
            values[i + 3] = uniform;
            values[i + 4] = uniform;
            values[i + 5] = uniform;
            values[i + 6] = uniform;
            values[i + 7] = uniform;
        }
        for (; i < length; i++) {
            values[i] = uniform;
        }
        return;
    }
    
    // Normalize in chunks
    for (i = 0; i < length - (CHUNK_SIZE - 1); i += CHUNK_SIZE) {
        int32_t scaled0 = ((int32_t)values[i] * q_scale) / sum;
        int32_t scaled1 = ((int32_t)values[i + 1] * q_scale) / sum;
        int32_t scaled2 = ((int32_t)values[i + 2] * q_scale) / sum;
        int32_t scaled3 = ((int32_t)values[i + 3] * q_scale) / sum;
        int32_t scaled4 = ((int32_t)values[i + 4] * q_scale) / sum;
        int32_t scaled5 = ((int32_t)values[i + 5] * q_scale) / sum;
        int32_t scaled6 = ((int32_t)values[i + 6] * q_scale) / sum;
        int32_t scaled7 = ((int32_t)values[i + 7] * q_scale) / sum;
        values[i] = (int16_t)((scaled0 < -32768) ? -32768 : 
                            (scaled0 > 32767) ? 32767 : scaled0);
        values[i + 1] = (int16_t)((scaled1 < -32768) ? -32768 : 
                                (scaled1 > 32767) ? 32767 : scaled1);
        values[i + 2] = (int16_t)((scaled2 < -32768) ? -32768 : 
                                (scaled2 > 32767) ? 32767 : scaled2);
        values[i + 3] = (int16_t)((scaled3 < -32768) ? -32768 : 
                                (scaled3 > 32767) ? 32767 : scaled3);
        values[i + 4] = (int16_t)((scaled4 < -32768) ? -32768 : 
                                (scaled4 > 32767) ? 32767 : scaled4);
        values[i + 5] = (int16_t)((scaled5 < -32768) ? -32768 : 
                                (scaled5 > 32767) ? 32767 : scaled5);
        values[i + 6] = (int16_t)((scaled6 < -32768) ? -32768 : 
                                (scaled6 > 32767) ? 32767 : scaled6);
        values[i + 7] = (int16_t)((scaled7 < -32768) ? -32768 : 
                                (scaled7 > 32767) ? 32767 : scaled7);
    }
    for (; i < length; i++) {
        int32_t scaled = ((int32_t)values[i] * q_scale) / sum;
        values[i] = (int16_t)((scaled < -32768) ? -32768 : 
                            (scaled > 32767) ? 32767 : scaled);
    }
}
*/
/*
void fake_softmax_S7_8(int16_t *values, int length) {
    // Constants for optimization
    const int16_t zero = 0;
    const int16_t max_exp = 32767;
    const int CHUNK_SIZE = 8;
    
    // Find max value in chunks
    int16_t max_val = values[0];
    int i;
    for(i = 1; i < length - (CHUNK_SIZE - 1); i += CHUNK_SIZE) {
        int16_t v0 = values[i];
        int16_t v1 = values[i + 1];
        int16_t v2 = values[i + 2];
        int16_t v3 = values[i + 3];
        int16_t v4 = values[i + 4];
        int16_t v5 = values[i + 5];
        int16_t v6 = values[i + 6];
        int16_t v7 = values[i + 7];
        
        if(v0 > max_val) max_val = v0;
        if(v1 > max_val) max_val = v1;
        if(v2 > max_val) max_val = v2;
        if(v3 > max_val) max_val = v3;
        if(v4 > max_val) max_val = v4;
        if(v5 > max_val) max_val = v5;
        if(v6 > max_val) max_val = v6;
        if(v7 > max_val) max_val = v7;
    }
    for(; i < length; i++) {
        if(values[i] > max_val) max_val = values[i];
    }
    
    // Pre-calculate exp table
    static const int16_t exp_table[] = {
        32767, 24100, 17720, 13030, 9580, 7040, 5180, 3810,
        2800, 2060, 1510, 1110, 820, 600, 440, 320, 240
    };
    
    // Compute exp and sum in single pass
    int32_t sum = 0;
    for(i = 0; i < length - (CHUNK_SIZE - 1); i += CHUNK_SIZE) {
        int16_t v0 = values[i] - max_val;
        int16_t v1 = values[i + 1] - max_val;
        int16_t v2 = values[i + 2] - max_val;
        int16_t v3 = values[i + 3] - max_val;
        int16_t v4 = values[i + 4] - max_val;
        int16_t v5 = values[i + 5] - max_val;
        int16_t v6 = values[i + 6] - max_val;
        int16_t v7 = values[i + 7] - max_val;
        
        // Branchless exp lookup
        v0 = (v0 < -8) ? zero : (v0 > 8) ? max_exp : exp_table[v0 + 8];
        v1 = (v1 < -8) ? zero : (v1 > 8) ? max_exp : exp_table[v1 + 8];
        v2 = (v2 < -8) ? zero : (v2 > 8) ? max_exp : exp_table[v2 + 8];
        v3 = (v3 < -8) ? zero : (v3 > 8) ? max_exp : exp_table[v3 + 8];
        v4 = (v4 < -8) ? zero : (v4 > 8) ? max_exp : exp_table[v4 + 8];
        v5 = (v5 < -8) ? zero : (v5 > 8) ? max_exp : exp_table[v5 + 8];
        v6 = (v6 < -8) ? zero : (v6 > 8) ? max_exp : exp_table[v6 + 8];
        v7 = (v7 < -8) ? zero : (v7 > 8) ? max_exp : exp_table[v7 + 8];
        
        values[i] = v0;
        values[i + 1] = v1;
        values[i + 2] = v2;
        values[i + 3] = v3;
        values[i + 4] = v4;
        values[i + 5] = v5;
        values[i + 6] = v6;
        values[i + 7] = v7;
        
        sum += v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
    }
    
    // Handle remaining elements
    for(; i < length; i++) {
        int16_t v = values[i] - max_val;
        v = (v < -8) ? zero : (v > 8) ? max_exp : exp_table[v + 8];
        values[i] = v;
        sum += v;
    }
    
    // Fast path for all zeros
    if(sum == 0) {
        int16_t uniform = (int16_t)(Q_SCALE / length);
        for(i = 0; i < length; i++) {
            values[i] = uniform;
        }
        return;
    }
    
    // Pre-calculate reciprocal for faster division
    int32_t reciprocal = (1LL << 32) / sum;
    
    // Normalize in chunks
    for(i = 0; i < length - (CHUNK_SIZE - 1); i += CHUNK_SIZE) {
        int64_t scaled0 = ((int64_t)values[i] * Q_SCALE * reciprocal) >> 32;
        int64_t scaled1 = ((int64_t)values[i + 1] * Q_SCALE * reciprocal) >> 32;
        int64_t scaled2 = ((int64_t)values[i + 2] * Q_SCALE * reciprocal) >> 32;
        int64_t scaled3 = ((int64_t)values[i + 3] * Q_SCALE * reciprocal) >> 32;
        int64_t scaled4 = ((int64_t)values[i + 4] * Q_SCALE * reciprocal) >> 32;
        int64_t scaled5 = ((int64_t)values[i + 5] * Q_SCALE * reciprocal) >> 32;
        int64_t scaled6 = ((int64_t)values[i + 6] * Q_SCALE * reciprocal) >> 32;
        int64_t scaled7 = ((int64_t)values[i + 7] * Q_SCALE * reciprocal) >> 32;
        
        // Branchless saturation
        values[i] = (int16_t)((scaled0 < -32768) ? -32768 : 
                            (scaled0 > 32767) ? 32767 : scaled0);
        values[i + 1] = (int16_t)((scaled1 < -32768) ? -32768 : 
                                (scaled1 > 32767) ? 32767 : scaled1);
        values[i + 2] = (int16_t)((scaled2 < -32768) ? -32768 : 
                                (scaled2 > 32767) ? 32767 : scaled2);
        values[i + 3] = (int16_t)((scaled3 < -32768) ? -32768 : 
                                (scaled3 > 32767) ? 32767 : scaled3);
        values[i + 4] = (int16_t)((scaled4 < -32768) ? -32768 : 
                                (scaled4 > 32767) ? 32767 : scaled4);
        values[i + 5] = (int16_t)((scaled5 < -32768) ? -32768 : 
                                (scaled5 > 32767) ? 32767 : scaled5);
        values[i + 6] = (int16_t)((scaled6 < -32768) ? -32768 : 
                                (scaled6 > 32767) ? 32767 : scaled6);
        values[i + 7] = (int16_t)((scaled7 < -32768) ? -32768 : 
                                (scaled7 > 32767) ? 32767 : scaled7);
    }
    
    // Handle remaining elements
    for(; i < length; i++) {
        int64_t scaled = ((int64_t)values[i] * Q_SCALE * reciprocal) >> 32;
        values[i] = (int16_t)((scaled < -32768) ? -32768 : 
                            (scaled > 32767) ? 32767 : scaled);
    }
}
*/
// ---------------------------------------------------------
// Single-head attention (int16 S7_8 version)
// Q, K, V: [SEQ_LEN][MODEL_DIM], all S7_8
// out_attn: [SEQ_LEN][MODEL_DIM], S7_8
// ---------------------------------------------------------
/*
void single_head_attention_S7_8(int16_t Q[SEQ_LEN][MODEL_DIM],
                              int16_t K[SEQ_LEN][MODEL_DIM],
                              int16_t V[SEQ_LEN][MODEL_DIM],
                              int16_t out_attn[SEQ_LEN][MODEL_DIM])
{
    // For each query i
    for(int i = 0; i < SEQ_LEN; i++){
        // compute attention scores vs each K[j]
        int16_t scores[SEQ_LEN];
        for(int j = 0; j < SEQ_LEN; j++){
            scores[j] = dot_S7_8(Q[i], K[j], MODEL_DIM);
        }

        // "Scale" by sqrt(MODEL_DIM) => in float code is / sqrt(4)=/2 => multiply by 0.5
        // in S7_8, multiplying by 0.5 means shifting by 1. So we do >> 1
        for(int j = 0; j < SEQ_LEN; j++){
            scores[j] = scores[j] >> 1; // approximate /2
        }

        // Fake softmax
        fake_softmax_S7_8(scores, SEQ_LEN);  // now each score in [0..Q_SCALE], sum=Q_SCALE

        // Weighted sum of V
        for(int d = 0; d < MODEL_DIM; d++){
            // accumulate in Q24, will shift to S7_8 at the end
            int32_t acc = 0;
            for(int j = 0; j < SEQ_LEN; j++){
                // scores[j] is S7_8, V[j][d] is S7_8 => product is Q16
                // we want final in S7_8 => sum of Q16 => shift down S7_8
                int32_t mul = (int32_t)scores[j] * (int32_t)V[j][d];
                acc += (mul >> Q_SHIFT); // now S7_8
            }
            out_attn[i][d] = saturate_i16(acc);
        }
    }
}*/
//29157 cycles
void single_head_attention_S7_8(int16_t Q[SEQ_LEN][MODEL_DIM],
                              int16_t K[SEQ_LEN][MODEL_DIM],
                              int16_t V[SEQ_LEN][MODEL_DIM],
                              int16_t out_attn[SEQ_LEN][MODEL_DIM])
{
    const int CHUNK_SIZE = 4;
    const int32_t q_shift = Q_SHIFT;
    
    // For each query i
    for(int i = 0; i < SEQ_LEN; i++) {
        int16_t scores[SEQ_LEN];
        for(int j = 0; j < SEQ_LEN - (CHUNK_SIZE - 1); j += CHUNK_SIZE) {
            scores[j] = dot_S7_8(Q[i], K[j], MODEL_DIM);
            scores[j + 1] = dot_S7_8(Q[i], K[j + 1], MODEL_DIM);
            scores[j + 2] = dot_S7_8(Q[i], K[j + 2], MODEL_DIM);
            scores[j + 3] = dot_S7_8(Q[i], K[j + 3], MODEL_DIM);
        }
        for(int j = SEQ_LEN - (SEQ_LEN % CHUNK_SIZE); j < SEQ_LEN; j++) {
            scores[j] = dot_S7_8(Q[i], K[j], MODEL_DIM);
        }
        for(int j = 0; j < SEQ_LEN - (CHUNK_SIZE - 1); j += CHUNK_SIZE) {
            scores[j] >>= 1;
            scores[j + 1] >>= 1;
            scores[j + 2] >>= 1;
            scores[j + 3] >>= 1;
        }
        for(int j = SEQ_LEN - (SEQ_LEN % CHUNK_SIZE); j < SEQ_LEN; j++) {
            scores[j] >>= 1;
        }

        // Fake softmax
        fake_softmax_S7_8(scores, SEQ_LEN);

        for(int d = 0; d < MODEL_DIM - (CHUNK_SIZE - 1); d += CHUNK_SIZE) {
            int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
            for(int j = 0; j < SEQ_LEN; j++) {
                int32_t score = scores[j];
                acc0 += (score * V[j][d]) >> q_shift;
                acc1 += (score * V[j][d + 1]) >> q_shift;
                acc2 += (score * V[j][d + 2]) >> q_shift;
                acc3 += (score * V[j][d + 3]) >> q_shift;
            }
            
            out_attn[i][d] = saturate_i16(acc0);
            out_attn[i][d + 1] = saturate_i16(acc1);
            out_attn[i][d + 2] = saturate_i16(acc2);
            out_attn[i][d + 3] = saturate_i16(acc3);
        }
        
        for(int d = MODEL_DIM - (MODEL_DIM % CHUNK_SIZE); d < MODEL_DIM; d++) {
            int32_t acc = 0;
            for(int j = 0; j < SEQ_LEN; j++) {
                acc += (scores[j] * V[j][d]) >> q_shift;
            }
            out_attn[i][d] = saturate_i16(acc);
        }
    }
}
//--------------------------------------------------------
//MHA
//--------------------------------------------------------
/*
void multi_head_attention_S7_8(int16_t Q[SEQ_LEN][MODEL_DIM],
                             int16_t K[SEQ_LEN][MODEL_DIM],
                             int16_t V[SEQ_LEN][MODEL_DIM],
                             int16_t out_attn[SEQ_LEN][MODEL_DIM])
{
    // Temporary storage for each head's output
    int16_t head_outputs[HEADS][SEQ_LEN][HEAD_DIM];
    
    // Process each head in parallel
    //[heads, seq_len, head_dim]
    //omp pragama parallel for
    for (int h = 0; h < HEADS; h++) {
        // Project inputs for this head
        int16_t Q_head[SEQ_LEN][HEAD_DIM];
        int16_t K_head[SEQ_LEN][HEAD_DIM];
        int16_t V_head[SEQ_LEN][HEAD_DIM];
        
        for (int i = 0; i < SEQ_LEN; i++) {
            matvec_mul_S7_8(WQ[h], Q[i], Q_head[i], HEAD_DIM, MODEL_DIM);
            matvec_mul_S7_8(WK[h], K[i], K_head[i], HEAD_DIM, MODEL_DIM);
            matvec_mul_S7_8(WV[h], V[i], V_head[i], HEAD_DIM, MODEL_DIM);
        }
        // SHA with S7_8 
        single_head_attention_S7_8(Q_head, K_head, V_head, head_outputs[h]);
    }
    
    // Combine head outputs
    // for every seq_len -> model dim 
    //get head idx, dimensions
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int d = 0; d < MODEL_DIM; d++) {
            int32_t acc = 0;
            int head_idx = d / HEAD_DIM;
            int dim_in_head = d % HEAD_DIM;
            acc = head_outputs[head_idx][i][dim_in_head];
            out_attn[i][d] = saturate_i16(acc);
        }
    }
}
*/

void multi_head_attention_S7_8(int16_t Q[SEQ_LEN][MODEL_DIM],
                             int16_t K[SEQ_LEN][MODEL_DIM],
                             int16_t V[SEQ_LEN][MODEL_DIM],
                             int16_t out_attn[SEQ_LEN][MODEL_DIM]){


    for (int h = 0; h < HEADS; h++) {
        
        int16_t Q_head[SEQ_LEN][HEAD_DIM];
        int16_t K_head[SEQ_LEN][HEAD_DIM];
        int16_t V_head[SEQ_LEN][HEAD_DIM];
        for (int i = 0; i < SEQ_LEN; i++) {
            matvec_mul_S7_8(WQ[h], Q[i], Q_head[i], HEAD_DIM, MODEL_DIM);
            matvec_mul_S7_8(WK[h], K[i], K_head[i], HEAD_DIM, MODEL_DIM);
            matvec_mul_S7_8(WV[h], V[i], V_head[i], HEAD_DIM, MODEL_DIM);
        }
        
        for(int i = 0; i < SEQ_LEN; i++){
            int16_t scores[SEQ_LEN];
            for(int j = 0; j < SEQ_LEN; j++){
                scores[j] = dot_S7_8(Q_head[i], K_head[j], HEAD_DIM);
                scores[j] = scores[j] >> 1;
            }
            // Apply softmax form sha
            fake_softmax_S7_8(scores, SEQ_LEN);
            for(int d = 0; d < HEAD_DIM; d++){
                int32_t acc = 0;
                for(int j = 0; j < SEQ_LEN; j++){
                    int32_t mul = (int32_t)scores[j] * (int32_t)V_head[j][d];
                    acc += (mul >> Q_SHIFT);
                }
                out_attn[i][h * HEAD_DIM + d] = saturate_i16(acc);
            }
        }
    }
}

//---------------------------------------------------------
//Layer normalization
//---------------------------------------------------------
// Add layer normalization function
// 2 degrees of momentum
int16_t gamma1[MODEL_DIM] = {
    TO_Q(0.0000001f), TO_Q(0.0000001f), TO_Q(0.0000001f), TO_Q(0.0000001f)
};
int16_t beta1[MODEL_DIM] = {
    TO_Q(0.0f), TO_Q(0.0f), TO_Q(0.0f), TO_Q(0.0f)
};
int16_t gamma2[MODEL_DIM] = {
    TO_Q(0.0000001f), TO_Q(0.0000001f), TO_Q(0.0000001f), TO_Q(0.0000001f)
};
int16_t beta2[MODEL_DIM] = {
    TO_Q(0.0f), TO_Q(0.0f), TO_Q(0.0f), TO_Q(0.0f)
};


void layer_norm_S7_8(int16_t input[SEQ_LEN][MODEL_DIM],
                    int16_t output[SEQ_LEN][MODEL_DIM],
                    int16_t gamma[MODEL_DIM],
                    int16_t beta[MODEL_DIM])
{
    //whiten results
    for(int i = 0; i < SEQ_LEN; i++){
        //mean
        int32_t sum = 0;
        for(int d = 0; d < MODEL_DIM; d++){
            sum += input[i][d];
        }
        int16_t mean = (int16_t)(sum / MODEL_DIM);
        // variance
        int32_t var_sum = 0;
        for(int d = 0; d < MODEL_DIM; d++){
            int32_t diff = (int32_t)input[i][d] - (int32_t)mean;
            var_sum += (diff * diff) >> Q_SHIFT;
        }
        int16_t variance = (int16_t)(var_sum / MODEL_DIM);
        int16_t epsilon = TO_Q(0.0001f);
        int16_t std_dev = (int16_t)sqrt((int32_t)variance + (int32_t)epsilon);
        // normalize shift 
        for(int d = 0; d < MODEL_DIM; d++){
            int32_t normalized = ((int32_t)input[i][d] - (int32_t)mean) * Q_SCALE / (int32_t)std_dev;
            int32_t scaled = ((int32_t)normalized * (int32_t)gamma[d]) >> Q_SHIFT;
            int32_t shifted = scaled + (int32_t)beta[d];
            output[i][d] = saturate_i16(shifted);
        }
    }
}


// ---------------------------------------------------------
// Feed-forward layer (2-layer MLP, ReLU in between)
// in_data, out_ff: [SEQ_LEN][MODEL_DIM], S7_8
// W1: [FF_DIM x MODEL_DIM], S7_8
// b1: [FF_DIM], S7_8
// W2: [MODEL_DIM x FF_DIM], S7_8
// b2: [MODEL_DIM], S7_8
// ---------------------------------------------------------
/*
void feed_forward_S7_8(int16_t in_data[SEQ_LEN][MODEL_DIM],
                      int16_t out_ff[SEQ_LEN][MODEL_DIM],
                      int16_t *W1,
                      int16_t *b1,
                      int16_t *W2,
                      int16_t *b2)
{
    for(int i = 0; i < SEQ_LEN; i++){
        // hidden = ReLU( in_data[i]*W1 + b1 )
        int16_t hidden[FF_DIM];
        matvec_mul_S7_8(W1, in_data[i], hidden, FF_DIM, MODEL_DIM);
        for(int h = 0; h < FF_DIM; h++){
            // add bias b1[h]
            int32_t sum = (int32_t)hidden[h] + (int32_t)b1[h];
            int16_t tmp = saturate_i16(sum);
            // ReLU
            if(tmp < 0) tmp = 0;
            hidden[h] = tmp;
        }

        // out_ff[i] = hidden * W2 + b2
        // hidden is [FF_DIM], W2 is [MODEL_DIM x FF_DIM]
        matvec_mul_S7_8(W2, hidden, out_ff[i], MODEL_DIM, FF_DIM);
        // add bias b2
        for(int d = 0; d < MODEL_DIM; d++){
            int32_t sum = (int32_t)out_ff[i][d] + (int32_t)b2[d];
            out_ff[i][d] = saturate_i16(sum);
        }
    }
}
*/
//28985
void feed_forward_S7_8(int16_t in_data[SEQ_LEN][MODEL_DIM],
                      int16_t out_ff[SEQ_LEN][MODEL_DIM],
                      int16_t *W1,
                      int16_t *b1,
                      int16_t *W2,
                      int16_t *b2)
{
    // Constants for optimization
    const int CHUNK_SIZE = 4;
    const int16_t zero = 0;
    
    for(int i = 0; i < SEQ_LEN; i++) {
        // hidden = ReLU( in_data[i]*W1 + b1 )
        int16_t hidden[FF_DIM];
        matvec_mul_S7_8(W1, in_data[i], hidden, FF_DIM, MODEL_DIM);
        
        // Process hidden layer in chunks
        for(int h = 0; h < FF_DIM - (CHUNK_SIZE - 1); h += CHUNK_SIZE) {
            // Process 4 elements at once
            int32_t sum0 = (int32_t)hidden[h] + (int32_t)b1[h];
            int32_t sum1 = (int32_t)hidden[h + 1] + (int32_t)b1[h + 1];
            int32_t sum2 = (int32_t)hidden[h + 2] + (int32_t)b1[h + 2];
            int32_t sum3 = (int32_t)hidden[h + 3] + (int32_t)b1[h + 3];
            
            // Branchless ReLU
            hidden[h] = (int16_t)((sum0 < 0) ? zero : 
                                (sum0 > 32767) ? 32767 : sum0);
            hidden[h + 1] = (int16_t)((sum1 < 0) ? zero : 
                                    (sum1 > 32767) ? 32767 : sum1);
            hidden[h + 2] = (int16_t)((sum2 < 0) ? zero : 
                                    (sum2 > 32767) ? 32767 : sum2);
            hidden[h + 3] = (int16_t)((sum3 < 0) ? zero : 
                                    (sum3 > 32767) ? 32767 : sum3);
        }
        
        // Handle remaining elements
        for(int h = FF_DIM - (FF_DIM % CHUNK_SIZE); h < FF_DIM; h++) {
            int32_t sum = (int32_t)hidden[h] + (int32_t)b1[h];
            hidden[h] = (int16_t)((sum < 0) ? zero : 
                                (sum > 32767) ? 32767 : sum);
        }

        // out_ff[i] = hidden * W2 + b2
        matvec_mul_S7_8(W2, hidden, out_ff[i], MODEL_DIM, FF_DIM);
        
        // Add bias b2 in chunks
        for(int d = 0; d < MODEL_DIM - (CHUNK_SIZE - 1); d += CHUNK_SIZE) {
            // Process 4 elements at once
            int32_t sum0 = (int32_t)out_ff[i][d] + (int32_t)b2[d];
            int32_t sum1 = (int32_t)out_ff[i][d + 1] + (int32_t)b2[d + 1];
            int32_t sum2 = (int32_t)out_ff[i][d + 2] + (int32_t)b2[d + 2];
            int32_t sum3 = (int32_t)out_ff[i][d + 3] + (int32_t)b2[d + 3];
            
            // Branchless saturation
            out_ff[i][d] = (int16_t)((sum0 < -32768) ? -32768 : 
                                    (sum0 > 32767) ? 32767 : sum0);
            out_ff[i][d + 1] = (int16_t)((sum1 < -32768) ? -32768 : 
                                        (sum1 > 32767) ? 32767 : sum1);
            out_ff[i][d + 2] = (int16_t)((sum2 < -32768) ? -32768 : 
                                        (sum2 > 32767) ? 32767 : sum2);
            out_ff[i][d + 3] = (int16_t)((sum3 < -32768) ? -32768 : 
                                        (sum3 > 32767) ? 32767 : sum3);
        }
        
        // Handle remaining elements
        for(int d = MODEL_DIM - (MODEL_DIM % CHUNK_SIZE); d < MODEL_DIM; d++) {
            int32_t sum = (int32_t)out_ff[i][d] + (int32_t)b2[d];
            out_ff[i][d] = (int16_t)((sum < -32768) ? -32768 : 
                                    (sum > 32767) ? 32767 : sum);
        }
    }
}
//28981 cycles
void run_transformer_encoder(int16_t final_out[SEQ_LEN][MODEL_DIM], int16_t volatile input[SEQ_LEN][MODEL_DIM])
{
    // Constants for optimization
    const int CHUNK_SIZE = 4;
    
    // -----------------------------------------------------
    // 1) Compute Q, K, V
    // -----------------------------------------------------
    int16_t Qmat[SEQ_LEN][MODEL_DIM];
    int16_t Kmat[SEQ_LEN][MODEL_DIM];
    int16_t Vmat[SEQ_LEN][MODEL_DIM];

    // Process input in chunks
    for(int i = 0; i < SEQ_LEN; i++) {
        matvec_mul_S7_8(WQ, input[i], Qmat[i], MODEL_DIM, MODEL_DIM);
        matvec_mul_S7_8(WK, input[i], Kmat[i], MODEL_DIM, MODEL_DIM);
        matvec_mul_S7_8(WV, input[i], Vmat[i], MODEL_DIM, MODEL_DIM);
    }

    // -----------------------------------------------------
    // 2) Single-head attention
    // -----------------------------------------------------
    int16_t attn_out[SEQ_LEN][MODEL_DIM];
    single_head_attention_S7_8(Qmat, Kmat, Vmat, attn_out);
    
    // -----------------------------------------------------
    // 3) Residual connection
    // -----------------------------------------------------
    int16_t post_attn[SEQ_LEN][MODEL_DIM];
    for(int i = 0; i < SEQ_LEN; i++) {
        // Process in chunks of 4
        for(int d = 0; d < MODEL_DIM - (CHUNK_SIZE - 1); d += CHUNK_SIZE) {
            // Process 4 elements at once
            int32_t sum0 = (int32_t)input[i][d] + (int32_t)attn_out[i][d];
            int32_t sum1 = (int32_t)input[i][d + 1] + (int32_t)attn_out[i][d + 1];
            int32_t sum2 = (int32_t)input[i][d + 2] + (int32_t)attn_out[i][d + 2];
            int32_t sum3 = (int32_t)input[i][d + 3] + (int32_t)attn_out[i][d + 3];
            
            // Branchless saturation
            post_attn[i][d] = (int16_t)((sum0 < -32768) ? -32768 : 
                                      (sum0 > 32767) ? 32767 : sum0);
            post_attn[i][d + 1] = (int16_t)((sum1 < -32768) ? -32768 : 
                                          (sum1 > 32767) ? 32767 : sum1);
            post_attn[i][d + 2] = (int16_t)((sum2 < -32768) ? -32768 : 
                                          (sum2 > 32767) ? 32767 : sum2);
            post_attn[i][d + 3] = (int16_t)((sum3 < -32768) ? -32768 : 
                                          (sum3 > 32767) ? 32767 : sum3);
        }
        
        // Handle remaining elements
        for(int d = MODEL_DIM - (MODEL_DIM % CHUNK_SIZE); d < MODEL_DIM; d++) {
            int32_t sum = (int32_t)input[i][d] + (int32_t)attn_out[i][d];
            post_attn[i][d] = (int16_t)((sum < -32768) ? -32768 : 
                                      (sum > 32767) ? 32767 : sum);
        }
    }
    
    // -----------------------------------------------------
    // 4) Feed-forward
    // -----------------------------------------------------
    int16_t ff_out[SEQ_LEN][MODEL_DIM];
    feed_forward_S7_8(post_attn, ff_out, W1, b1, W2, b2);
    
    // -----------------------------------------------------
    // 5) Final residual connection
    // -----------------------------------------------------
    for(int i = 0; i < SEQ_LEN; i++) {
        // Process in chunks of 4
        for(int d = 0; d < MODEL_DIM - (CHUNK_SIZE - 1); d += CHUNK_SIZE) {
            // Process 4 elements at once
            int32_t sum0 = (int32_t)post_attn[i][d] + (int32_t)ff_out[i][d];
            int32_t sum1 = (int32_t)post_attn[i][d + 1] + (int32_t)ff_out[i][d + 1];
            int32_t sum2 = (int32_t)post_attn[i][d + 2] + (int32_t)ff_out[i][d + 2];
            int32_t sum3 = (int32_t)post_attn[i][d + 3] + (int32_t)ff_out[i][d + 3];
            
            // Branchless saturation
            final_out[i][d] = (int16_t)((sum0 < -32768) ? -32768 : 
                                      (sum0 > 32767) ? 32767 : sum0);
            final_out[i][d + 1] = (int16_t)((sum1 < -32768) ? -32768 : 
                                          (sum1 > 32767) ? 32767 : sum1);
            final_out[i][d + 2] = (int16_t)((sum2 < -32768) ? -32768 : 
                                          (sum2 > 32767) ? 32767 : sum2);
            final_out[i][d + 3] = (int16_t)((sum3 < -32768) ? -32768 : 
                                          (sum3 > 32767) ? 32767 : sum3);
        }
        
        // Handle remaining elements
        for(int d = MODEL_DIM - (MODEL_DIM % CHUNK_SIZE); d < MODEL_DIM; d++) {
            int32_t sum = (int32_t)post_attn[i][d] + (int32_t)ff_out[i][d];
            final_out[i][d] = (int16_t)((sum < -32768) ? -32768 : 
                                      (sum > 32767) ? 32767 : sum);
        }
    }
}

/*
void run_transformer_encoder(int16_t final_out[SEQ_LEN][MODEL_DIM] ,int16_t volatile input[SEQ_LEN][MODEL_DIM])
{
        // -----------------------------------------------------
    // 1) Compute Q, K, V
    // -----------------------------------------------------
    int16_t Qmat[SEQ_LEN][MODEL_DIM];
    int16_t Kmat[SEQ_LEN][MODEL_DIM];
    int16_t Vmat[SEQ_LEN][MODEL_DIM];

    for(int i = 0; i < SEQ_LEN; i++){
        //for(int h=0;h < HEADS; h++){
         For MHA
        matvec_mul_S7_8(WQ[h], input[i], Qmat[i][h * HEAD_DIM], MODEL_DIM, MODEL_DIM);
        matvec_mul_S7_8(WK[h], input[i], Kmat[i][h * HEAD_DIM], MODEL_DIM, MODEL_DIM);
        matvec_mul_S7_8(WV[h], input[i], Vmat[i][h * HEAD_DIM], MODEL_DIM, MODEL_DIM);
       } 
        matvec_mul_S7_8(WQ, input[i], Qmat[i], MODEL_DIM, MODEL_DIM);
        matvec_mul_S7_8(WK, input[i], Kmat[i], MODEL_DIM, MODEL_DIM);
        matvec_mul_S7_8(WV, input[i], Vmat[i], MODEL_DIM, MODEL_DIM);
       
    }

    // -----------------------------------------------------
    // 2) Single-head attention
    // -----------------------------------------------------
    int16_t attn_out[SEQ_LEN][MODEL_DIM];
    single_head_attention_S7_8(Qmat, Kmat, Vmat, attn_out);
    
    // -----------------------------------------------------
    // 2) Multi-head attention
    // -----------------------------------------------------
    //int16_t attn_out[SEQ_LEN][MODEL_DIM];
    //multi_head_attention_S7_8(Qmat, Kmat, Vmat, attn_out); 

    
    // Residual (skip true layer norm for brevity)
    int16_t post_attn[SEQ_LEN][MODEL_DIM];
    for(int i = 0; i < SEQ_LEN; i++){
        for(int d = 0; d < MODEL_DIM; d++){
            int32_t sum = (int32_t)input[i][d] + (int32_t)attn_out[i][d];
            post_attn[i][d] = saturate_i16(sum);
        }
    }

    // -----------------------------------------------------
    // 3) Layer norm
    // -----------------------------------------------------
    
    //int16_t norm_attn[SEQ_LEN][MODEL_DIM];
    //layer_norm_S7_8(post_attn, norm_attn, gamma1, beta1);

    // -----------------------------------------------------
    // 4) Feed-forward
    // -----------------------------------------------------
    int16_t ff_out[SEQ_LEN][MODEL_DIM];
    feed_forward_S7_8(post_attn, ff_out, W1, b1, W2, b2);
    //feed_forward_S7_8(norm_attn, ff_out, W1, b1, W2, b2);
    
    // Residual again
    
    for(int i = 0; i < SEQ_LEN; i++){
        for(int d = 0; d < MODEL_DIM; d++){
            int32_t sum = (int32_t)post_attn[i][d] + (int32_t)ff_out[i][d];
            final_out[i][d] = saturate_i16(sum);
        }
    }
    
    // Residual final with norm_attn
    int16_t post_ff[SEQ_LEN][MODEL_DIM];
    for(int i = 0; i < SEQ_LEN; i++){
        for(int d = 0; d < MODEL_DIM; d++){
            int32_t sum = (int32_t)norm_attn[i][d] + (int32_t)ff_out[i][d];
            //post_ff[i][d] = saturate_i16(sum);
            final_out[i][d] = saturate_i16(sum);
        }
    }
    // -----------------------------------------------------
    // 3) Layer-Norm final
    // -----------------------------------------------------
    //layer_norm_S7_8(post_ff, final_out, gamma2, beta2);
    
    return;
}*/
//=======================================================================================================
// END OF HACKATHON CODE
//=======================================================================================================
//=======================================================================================================
//=======================================================================================================
//=======================================================================================================
//=======================================================================================================



// ---------------------------------------------------------
// Main Demo using s7_8 variables, i.e., fixed-point with 1 sign bit, 7 integer bits, and 8 fractional bits
// ---------------------------------------------------------
int main(void) {
    uint32_t exectime = 0, a,b,t;

    // Precalculated embeddings
    // Example input: 3 tokens, each dimension=4, stored in S7_8
    int16_t volatile input[SEQ_LEN][MODEL_DIM] = {
        { TO_Q(0.1f), TO_Q(0.2f), TO_Q(0.3f), TO_Q(0.4f) },
        { TO_Q(0.2f), TO_Q(0.1f), TO_Q(0.5f), TO_Q(0.3f) },
        { TO_Q(0.5f), TO_Q(0.9f), TO_Q(0.1f), TO_Q(0.0f) }
    };


    a = GETCPUTIME();

    int16_t final_out[SEQ_LEN][MODEL_DIM];
    run_transformer_encoder(final_out, input);

    b = GETCPUTIME();
    t = b-a;
    if (t<0) {
    t += 0x7fffffff;
    }
    exectime += t;

    // -----------------------------------------------------
    // Finalize
    // -----------------------------------------------------
    int32_t reference_result[12] = {168, 210, 225, 270,195, 189, 284, 250,293, 414, 193, 201};
    int32_t check = 0;
    printf("== Final Transformer Output (S7_8) ==\n");
    for(int i = 0; i < SEQ_LEN; i++){
        printf("Token %d: [", i);
        for(int d = 0; d < MODEL_DIM; d++){
            //print_S7_8(final_out[i][d]);
            check += (final_out[i][d]-reference_result[d+i*MODEL_DIM])*(final_out[i][d]-reference_result[d+i*MODEL_DIM]);
            printf("%d",final_out[i][d]);
            if(d < MODEL_DIM - 1) printf(", ");
        }
        printf(" ]\n");
    }
    printf("\n== Verification ==\n");
    if (!check)
        printf("PASSED!\n");
    else
        printf("FAILED!\n");
    printf("\n== Performance ==\n");
    printf("Cycles = %d\n",exectime);

    return 0;
}
