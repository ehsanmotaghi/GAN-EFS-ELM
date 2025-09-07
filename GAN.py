import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate  # pip install tabulate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers
import os
import time
import datetime

from sklearn.utils import resample
from tensorflow.keras.utils import to_categorical

from scipy.stats import ks_2samp, entropy
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

# Replace two points (..) with the file path.

data_simulator1 = pd.read_csv('../No-GAN_No-EFS(resampled_39).csv')
data_simulator1.shape   # (3174, 39)

##########
# Check the count of category class
counts = data_simulator1.value_counts('type')
print(counts)
# let's see the distribution of our target category by bar chart
plt.figure(figsize=(10,5))
plt.xticks(rotation=45, ha='right')
ax = sns.countplot(x = 'type', data = data_simulator1, order=data_simulator1['type'].value_counts(ascending=False).index, palette = 'Set2')
for container in ax.containers:
    ax.bar_label(container)
ax.set_title("Count of Normal and Attack Types")
##########

df = data_simulator1
X = df.drop(columns=['label', 'type'])
X.shape
y = df['type']
target_col = y

##########
# Utility: Preprocessing (scaling, encoding)
def preprocess_data(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X_scaled, y_enc, scaler, le

def postprocess_data(X, y, scaler, le, columns, target_col):
    X_inv = scaler.inverse_transform(X)
    y_inv = le.inverse_transform(y)
    df = pd.DataFrame(X_inv, columns=columns)
    df[target_col] = y_inv
    return df

##########
# Vanilla GAN
class VanillaGAN:
    def __init__(self, input_dim, latent_dim=32):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

    # def build_generator(self):
    #     model = tf.keras.Sequential([
    #         layers.Dense(64, activation='relu', input_dim=self.latent_dim),
    #         layers.Dense(128, activation='relu'),
    #         layers.Dense(self.input_dim, activation='linear')
    #     ])
    #     return model
    
    def build_generator(self):
        model = tf.keras.Sequential([
            layers.Dense(64, activation='LeakyReLU', input_dim=self.latent_dim),
            layers.Dense(128, activation='LeakyReLU'),
            layers.Dense(self.input_dim, activation='sigmoid')
        ])
        return model

    # def build_discriminator(self):
    #     model = tf.keras.Sequential([
    #         layers.Dense(128, activation='relu', input_dim=self.input_dim),
    #         layers.Dense(64, activation='relu'),
    #         layers.Dense(1, activation='sigmoid')
    #     ])
    #     model.compile(optimizer='adam', loss='binary_crossentropy')
    #     return model
    
    def build_discriminator(self):
        model = tf.keras.Sequential([
            layers.Dense(128, activation='LeakyReLU', input_dim=self.input_dim),
            layers.Dense(64, activation='LeakyReLU'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def build_gan(self):
        self.discriminator.trainable = False
        model = tf.keras.Sequential([
            self.generator,
            self.discriminator
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def train(self, X, epochs=1000, batch_size=32):
        start_time = time.time()
        for epoch in range(epochs):
            # Train discriminator
            idx = np.random.randint(0, X.shape[0], batch_size)
            real = X[idx]
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake = self.generator.predict(noise, verbose=0)
            X_combined = np.vstack([real, fake])
            y_combined = np.hstack([np.ones(batch_size), np.zeros(batch_size)])
            d_loss = self.discriminator.train_on_batch(X_combined, y_combined)
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            y_gen = np.ones(batch_size)
            g_loss = self.gan.train_on_batch(noise, y_gen)
        elapsed = time.time() - start_time
        print(f"VanillaGAN training time: {elapsed:.2f} seconds ({elapsed/60:.2f} min)")
        return self, elapsed

    def generate(self, n_samples):
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        return self.generator.predict(noise, verbose=0)
   
       
##########

# Encode label column: all non-zero to 1, zero stays 0
# if 'label' in df.columns:
#     df['label'] = (df['label'] != 0).astype(int)

# # Apply MinMax scaling to all columns except 'label' and 'type'
# cols_to_scale = [col for col in df.columns if col not in ['label', 'type']]
# if cols_to_scale:
#     scaler = MinMaxScaler()
#     df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

##########
# WGAN
class WGAN:
    def __init__(self, input_dim, latent_dim=32, clip_value=0.01):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.clip_value = clip_value
        self.generator = self.build_generator()
        self.critic = self.build_critic()
        self.gan = self.build_gan()

    def build_generator(self):
        model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_dim=self.latent_dim),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.input_dim, activation='linear')
        ])
        return model

    def build_critic(self):
        model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=self.input_dim),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00005), loss=self.wasserstein_loss)
        return model

    def build_gan(self):
        self.critic.trainable = False
        model = tf.keras.Sequential([
            self.generator,
            self.critic
        ])
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00005), loss=self.wasserstein_loss)
        return model

    def wasserstein_loss(self, y_true, y_pred):
        return tf.reduce_mean(y_true * y_pred)

    def train(self, X, epochs=1000, batch_size=32, n_critic=5):
        start_time = time.time()
        for epoch in range(epochs):
            for _ in range(n_critic):
                idx = np.random.randint(0, X.shape[0], batch_size)
                real = X[idx]
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                fake = self.generator.predict(noise, verbose=0)
                X_combined = np.vstack([real, fake])
                y_combined = np.hstack([np.ones(batch_size), -np.ones(batch_size)])
                d_loss = self.critic.train_on_batch(X_combined, y_combined)
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            y_gen = np.ones(batch_size)
            g_loss = self.gan.train_on_batch(noise, y_gen)
        elapsed = time.time() - start_time
        print(f"WGAN training time: {elapsed:.2f} seconds ({elapsed/60:.2f} min)")
        return self, elapsed

    def generate(self, n_samples):
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        return self.generator.predict(noise, verbose=0)

##########
# WGAN-GP
class WGANGP:
    def __init__(self, input_dim, latent_dim=32, gp_weight=10):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight
        self.generator = self.build_generator()
        self.critic = self.build_critic()
        self.gan = self.build_gan()

    def build_generator(self):
        model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_dim=self.latent_dim),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.input_dim, activation='linear')
        ])
        return model

    def build_critic(self):
        model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=self.input_dim),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss=self.wasserstein_loss)
        return model

    def build_gan(self):
        self.critic.trainable = False
        model = tf.keras.Sequential([
            self.generator,
            self.critic
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss=self.wasserstein_loss)
        return model

    def wasserstein_loss(self, y_true, y_pred):
        return tf.reduce_mean(y_true * y_pred)

    def gradient_penalty(self, real, fake):
        alpha = tf.random.uniform([real.shape[0], 1], 0., 1.)
        interpolated = alpha * real + (1 - alpha) * fake
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.critic(interpolated)
        grads = tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
        gp = tf.reduce_mean((norm - 1.) ** 2)
        return gp

    def train(self, X, epochs=1000, batch_size=32, n_critic=5):
        start_time = time.time()
        for epoch in range(epochs):
            for _ in range(n_critic):
                idx = np.random.randint(0, X.shape[0], batch_size)
                real = X[idx]
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                fake = self.generator.predict(noise, verbose=0)
                with tf.GradientTape() as tape:
                    d_real = self.critic(real)
                    d_fake = self.critic(fake)
                    gp = self.gradient_penalty(real, fake)
                    d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real) + self.gp_weight * gp
                grads = tape.gradient(d_loss, self.critic.trainable_variables)
                self.critic.optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                fake = self.generator(noise)
                g_loss = -tf.reduce_mean(self.critic(fake))
            grads = tape.gradient(g_loss, self.generator.trainable_variables)
            self.gan.optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        elapsed = time.time() - start_time
        print(f"WGANGP training time: {elapsed:.2f} seconds ({elapsed/60:.2f} min)")
        return self, elapsed

    def generate(self, n_samples):
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        return self.generator.predict(noise, verbose=0)

##########
# CGAN
class CGAN:
    def __init__(self, input_dim, n_classes, latent_dim=32):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

    def build_generator(self):
        noise = layers.Input(shape=(self.latent_dim,))
        label = layers.Input(shape=(self.n_classes,))
        x = layers.Concatenate()([noise, label])
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        out = layers.Dense(self.input_dim, activation='linear')(x)
        return tf.keras.Model([noise, label], out)

    def build_discriminator(self):
        data = layers.Input(shape=(self.input_dim,))
        label = layers.Input(shape=(self.n_classes,))
        x = layers.Concatenate()([data, label])
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        out = layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model([data, label], out)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def build_gan(self):
        self.discriminator.trainable = False
        noise = layers.Input(shape=(self.latent_dim,))
        label = layers.Input(shape=(self.n_classes,))
        gen_out = self.generator([noise, label])
        gan_out = self.discriminator([gen_out, label])
        model = tf.keras.Model([noise, label], gan_out)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def train(self, X, y, epochs=1000, batch_size=32):
        start_time = time.time()
        y_cat = to_categorical(y, num_classes=self.n_classes)
        for epoch in range(epochs):
            idx = np.random.randint(0, X.shape[0], batch_size)
            real = X[idx]
            real_labels = y_cat[idx]
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_labels = y_cat[np.random.randint(0, X.shape[0], batch_size)]
            fake = self.generator.predict([noise, fake_labels], verbose=0)
            X_combined = np.vstack([real, fake])
            y_combined = np.vstack([real_labels, fake_labels])
            d_y = np.hstack([np.ones(batch_size), np.zeros(batch_size)])
            d_loss = self.discriminator.train_on_batch([X_combined, y_combined], d_y)
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            sampled_labels = y_cat[np.random.randint(0, X.shape[0], batch_size)]
            g_loss = self.gan.train_on_batch([noise, sampled_labels], np.ones(batch_size))
        elapsed = time.time() - start_time
        print(f"CGAN training time: {elapsed:.2f} seconds ({elapsed/60:.2f} min)")
        return self, elapsed

    def generate(self, n_samples, class_idx):
        noise = np.random.normal(0, 1, (n_samples, self.latent_dim))
        labels = np.zeros((n_samples, self.n_classes))
        labels[:, class_idx] = 1
        return self.generator.predict([noise, labels], verbose=0)

##########
# Main oversampling function
def oversample_with_gan(df, target_col, gan_type='vanilla', n_samples_per_class=None, epochs=1000, batch_size=32):
    X, y, scaler, le = preprocess_data(df, target_col)
    columns = df.drop(columns=[target_col]).columns
    class_counts = np.bincount(y)
    max_count = np.max(class_counts)
    n_classes = len(class_counts)
    if n_samples_per_class is None:
        n_samples_per_class = max_count
    synthetic_X = []
    synthetic_y = []
    timing_info = {}
    for class_idx in range(n_classes):
        n_to_generate = n_samples_per_class - class_counts[class_idx]
        if n_to_generate <= 0:
            continue
        X_class = X[y == class_idx]
        if gan_type == 'vanilla':
            gan = VanillaGAN(input_dim=X.shape[1])
            _, elapsed = gan.train(X_class, epochs=epochs, batch_size=batch_size)
            timing_info[f'class_{class_idx}_vanilla'] = elapsed
            X_fake = gan.generate(n_to_generate)
        elif gan_type == 'wgan':
            gan = WGAN(input_dim=X.shape[1])
            _, elapsed = gan.train(X_class, epochs=epochs, batch_size=batch_size)
            timing_info[f'class_{class_idx}_wgan'] = elapsed
            X_fake = gan.generate(n_to_generate)
        elif gan_type == 'wgangp':
            gan = WGANGP(input_dim=X.shape[1])
            _, elapsed = gan.train(X_class, epochs=epochs, batch_size=batch_size)
            timing_info[f'class_{class_idx}_wgangp'] = elapsed
            X_fake = gan.generate(n_to_generate)
        elif gan_type == 'cgan':
            gan = CGAN(input_dim=X.shape[1], n_classes=n_classes)
            _, elapsed = gan.train(X, y, epochs=epochs, batch_size=batch_size)
            timing_info[f'class_{class_idx}_cgan'] = elapsed
            X_fake = gan.generate(n_to_generate, class_idx)
        else:
            raise ValueError(f"Unknown GAN type: {gan_type}")
        synthetic_X.append(X_fake)
        synthetic_y.append(np.full(n_to_generate, class_idx))
    if synthetic_X:
        X_syn = np.vstack(synthetic_X)
        y_syn = np.hstack(synthetic_y)
        df_syn = postprocess_data(X_syn, y_syn, scaler, le, columns, target_col)
        df_out = pd.concat([df, df_syn], ignore_index=True)
    else:
        df_out = df.copy()
    print(f"GAN training timing info: {timing_info}")
    return df_out, timing_info

##########
# --- Synthetic Data Quality Evaluation Functions ---
def evaluate_statistical_similarity(real_df, synth_df, target_col):
    ks_results = {}
    js_results = {}
    for col in real_df.columns:
        if col == target_col:
            continue
        real = real_df[col].values
        synth = synth_df[col].values
        # KS test
        try:
            ks_stat, ks_p = ks_2samp(real, synth)
        except Exception:
            ks_stat, ks_p = None, None
        ks_results[col] = {'ks_stat': ks_stat, 'ks_p': ks_p}
        # JS divergence (histogram-based)
        try:
            real_hist, _ = np.histogram(real, bins=20, density=True)
            synth_hist, _ = np.histogram(synth, bins=20, density=True)
            # Add small value to avoid log(0)
            real_hist += 1e-8
            synth_hist += 1e-8
            m = 0.5 * (real_hist + synth_hist)
            js = 0.5 * (entropy(real_hist, m) + entropy(synth_hist, m))
        except Exception:
            js = None
        js_results[col] = js
    return ks_results, js_results

def evaluate_class_distribution(real_df, synth_df, target_col):
    real_dist = real_df[target_col].value_counts(normalize=True)
    synth_dist = synth_df[target_col].value_counts(normalize=True)
    return real_dist, synth_dist

def evaluate_classifier(real_df, synth_df, target_col, classifier='all', gan_variant=None):
    # Label: 0 for real, 1 for synthetic
    df_real = real_df.copy()
    df_real['__label__'] = 0
    df_synth = synth_df.copy()
    df_synth['__label__'] = 1
    df_all = pd.concat([df_real, df_synth], ignore_index=True)
    X = df_all.drop(columns=[target_col, '__label__'])
    y = df_all['__label__']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    results = {}
    classifier = classifier.lower()
   
    # Random Forest
    if classifier in ['all', 'randomforest']:
        clf_rf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf_rf.fit(X_train, y_train)
        y_pred_rf = clf_rf.predict(X_test)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
        results['RandomForest'] = {'accuracy': acc_rf, 'report': report_rf}
    
    # XGBoost (if available)
    if classifier in ['all', 'xgboost']:
        try:
            # from xgboost import XGBClassifier
            clf_xgb = XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss', random_state=42, verbosity=0)
            clf_xgb.fit(X_train, y_train)
            y_pred_xgb = clf_xgb.predict(X_test)
            acc_xgb = accuracy_score(y_test, y_pred_xgb)
            report_xgb = classification_report(y_test, y_pred_xgb, output_dict=True)
            results['XGBoost'] = {'accuracy': acc_xgb, 'report': report_xgb}
            # Print XGBoost metrics as a table
            # import matplotlib.pyplot as plt
            # import datetime
            report_xgb_df = pd.DataFrame(report_xgb).transpose()
            table_title = "\nXGBoost Evaluation Metrics Table:"
            if gan_variant:
                table_title += f" ({gan_variant})"
            print(table_title)
            try:
                from tabulate import tabulate
                print(tabulate(report_xgb_df, headers='keys', tablefmt='psql', showindex=True))
            except ImportError:
                print(report_xgb_df)
            # Save as CSV and image
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = f"xgboost_metrics_{timestamp}.csv"
            img_path = f"xgboost_metrics_{timestamp}.png"
            report_xgb_df.to_csv(csv_path)
            # Save as image
            fig, ax = plt.subplots(figsize=(10, 2 + 0.5 * len(report_xgb_df)))
            ax.axis('off')
            tbl = ax.table(cellText=report_xgb_df.values, colLabels=report_xgb_df.columns, rowLabels=report_xgb_df.index, loc='center', cellLoc='center')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(12)
            tbl.auto_set_column_width(col=list(range(len(report_xgb_df.columns))))
            plt.title('XGBoost Evaluation Metrics', fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(img_path, bbox_inches='tight')
            plt.close(fig)
        except ImportError:
            results['XGBoost'] = None
    # KNN Classifier
    if classifier in ['all', 'knn']:
        try:
            # from sklearn.neighbors import KNeighborsClassifier
            clf_knn = KNeighborsClassifier(n_neighbors=5)
            clf_knn.fit(X_train, y_train)
            y_pred_knn = clf_knn.predict(X_test)
            acc_knn = accuracy_score(y_test, y_pred_knn)
            report_knn = classification_report(y_test, y_pred_knn, output_dict=True)
            results['KNN'] = {'accuracy': acc_knn, 'report': report_knn}
        except ImportError:
            results['KNN'] = None
    return results


##########
# --- Automated Reporting Function ---
def synthetic_data_report(real_df, synth_df, target_col, gan_type, print_report=True, save_path=None):
    ks_results, js_results = evaluate_statistical_similarity(real_df, synth_df, target_col)
    real_dist, synth_dist = evaluate_class_distribution(real_df, synth_df, target_col)
    clf_results = evaluate_classifier(real_df, synth_df, target_col, gan_variant=gan_type)
    report_lines = []
    report_lines.append(f"\n===== Synthetic Data Quality Report: {gan_type.upper()} =====\n")
    report_lines.append("--- Statistical Similarity (KS test, JS divergence) ---")
    for col in ks_results:
        ks = ks_results[col]
        js = js_results[col]
        report_lines.append(f"Feature: {col} | KS-stat: {ks['ks_stat']:.4f} (p={ks['ks_p']:.4f}) | JS-div: {js:.4f}")
    report_lines.append("\n--- Class Distribution (real vs. synthetic) ---")
    report_lines.append(f"Real: {real_dist.to_dict()}")
    report_lines.append(f"Synthetic: {synth_dist.to_dict()}")
    report_lines.append("\n--- Classifier-based Evaluation ---")
    # Random Forest
    if 'RandomForest' in clf_results and clf_results['RandomForest'] is not None:
        acc_rf = clf_results['RandomForest']['accuracy']
        report_rf = clf_results['RandomForest']['report']
        report_lines.append(f"Random Forest Accuracy (real vs. synthetic): {acc_rf:.4f}")
        report_lines.append(f"Random Forest Classification Report: {report_rf}")
    # XGBoost
    if 'XGBoost' in clf_results:
        if clf_results['XGBoost'] is not None:
            acc_xgb = clf_results['XGBoost']['accuracy']
            report_xgb = clf_results['XGBoost']['report']
            report_lines.append(f"XGBoost Accuracy (real vs. synthetic): {acc_xgb:.4f}")
            report_lines.append(f"XGBoost Classification Report: {report_xgb}")
        else:
            report_lines.append("XGBoost not available (xgboost not installed)")
    # KNN
    if 'KNN' in clf_results:
        if clf_results['KNN'] is not None:
            acc_knn = clf_results['KNN']['accuracy']
            report_knn = clf_results['KNN']['report']
            report_lines.append(f"KNN Accuracy (real vs. synthetic): {acc_knn:.4f}")
            report_lines.append(f"KNN Classification Report: {report_knn}")
        else:
            report_lines.append("KNN not available (sklearn.neighbors not installed)")
    report = '\n'.join(report_lines)
    if print_report:
        print(report)
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)

    # Visualization section
    if print_report:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        # 1. KS and JS bar plots
        ks_stats = {k: v['ks_stat'] for k, v in ks_results.items()}
        js_stats = js_results
        features = list(ks_stats.keys())
        plt.figure(figsize=(12, 5))
        plt.bar(features, [ks_stats[f] for f in features], color='skyblue')
        plt.title(f'KS Statistic per Feature ({gan_type})')
        plt.ylabel('KS Statistic')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(12, 5))
        plt.bar(features, [js_stats[f] for f in features], color='salmon')
        plt.title(f'JS Divergence per Feature ({gan_type})')
        plt.ylabel('JS Divergence')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        # 2. Class distribution bar plot
        dist_df = pd.DataFrame({'Real': real_dist, 'Synthetic': synth_dist}).fillna(0)
        dist_df.plot(kind='bar', figsize=(10,5))
        plt.title(f'Class Distribution: Real vs Synthetic ({gan_type})')
        plt.ylabel('Proportion')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        # 3. Classifier accuracy bar plot (Random Forest, XGBoost, KNN if available)
        accs = []
        names = []
        if 'RandomForest' in clf_results and clf_results['RandomForest'] is not None:
            accs.append(clf_results['RandomForest']['accuracy'])
            names.append('RandomForest')
        if 'XGBoost' in clf_results and clf_results['XGBoost'] is not None:
            accs.append(clf_results['XGBoost']['accuracy'])
            names.append('XGBoost')
        if 'KNN' in clf_results and clf_results['KNN'] is not None:
            accs.append(clf_results['KNN']['accuracy'])
            names.append('KNN')
        if accs:
            plt.figure(figsize=(6,4))
            plt.bar(names, accs, color=['mediumseagreen','orange','dodgerblue'][:len(accs)])
            plt.ylim(0, 1)
            plt.title(f'Classifier Accuracy (Real vs Synthetic) ({gan_type})')
            plt.ylabel('Accuracy')
            plt.tight_layout()
            plt.show()
    return report

##########
def plot_tsne_comparison_by_target(real_df, synth_df, target_col, n_samples=1000, random_state=42, ax=None, title=None, gan_variant=None):
    # Sample to speed up t-SNE if data is large
    real_sample = real_df.sample(n=min(n_samples, len(real_df)), random_state=random_state)
    synth_sample = synth_df.sample(n=min(n_samples, len(synth_df)), random_state=random_state)
    # Add source label
    real_sample = real_sample.copy()
    synth_sample = synth_sample.copy()
    real_sample['__source__'] = 'Real'
    synth_sample['__source__'] = 'Synthetic'
    # Combine
    combined = pd.concat([real_sample, synth_sample], ignore_index=True)
    # Only use numeric columns for t-SNE
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    features = combined[numeric_cols]
    # t-SNE
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30)
    tsne_results = tsne.fit_transform(features)
    combined['tsne-1'] = tsne_results[:,0]
    combined['tsne-2'] = tsne_results[:,1]
    # Plot
    if ax is None:
        plt.figure(figsize=(10,6))
        ax = plt.gca()
    plot_title = title if title else 't-SNE: Real vs Synthetic Data'
    if gan_variant:
        plot_title += f' ({gan_variant})'
    sns.scatterplot(
        x='tsne-1', y='tsne-2',
        hue='__source__',
        style=target_col if target_col in combined.columns else None,
        data=combined,
        alpha=0.7,
        palette='Set1',
        ax=ax
    )
    ax.set_title(plot_title)
    ax.legend()

def plot_tsne_comparison(real_df, synth_df, n_samples=1000, random_state=42, gan_variant=None):
    # Sample to speed up t-SNE if data is large
    real_sample = real_df.sample(n=min(n_samples, len(real_df)), random_state=random_state)
    synth_sample = synth_df.sample(n=min(n_samples, len(synth_df)), random_state=random_state)
    # Add source label
    real_sample = real_sample.copy()
    synth_sample = synth_sample.copy()
    real_sample['__source__'] = 'Real'
    synth_sample['__source__'] = 'Synthetic'
    # Combine
    combined = pd.concat([real_sample, synth_sample], ignore_index=True)
    # Only use numeric columns for t-SNE
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    features = combined[numeric_cols]
    # t-SNE
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30)
    tsne_results = tsne.fit_transform(features)
    combined['tsne-1'] = tsne_results[:,0]
    combined['tsne-2'] = tsne_results[:,1]
    # Plot
    plt.figure(figsize=(10,6))
    plot_title = 't-SNE: Real vs Synthetic Data'
    if gan_variant:
        plot_title += f' ({gan_variant})'
    sns.scatterplot(
        x='tsne-1', y='tsne-2',
        hue='__source__',
        data=combined,
        alpha=0.7,
        palette='Set1'
    )
    plt.title(plot_title)
    plt.legend()
    plt.show()

##########
def plot_tsne_comparison_3d(real_df, synth_df, target_col=None, n_samples=1000, random_state=42, gan_variant=None):
    # Sample to speed up t-SNE if data is large
    real_sample = real_df.sample(n=min(n_samples, len(real_df)), random_state=random_state)
    synth_sample = synth_df.sample(n=min(n_samples, len(synth_df)), random_state=random_state)
    # Add source label
    real_sample = real_sample.copy()
    synth_sample = synth_sample.copy()
    real_sample['__source__'] = 'Real'
    synth_sample['__source__'] = 'Synthetic'
    # Combine
    combined = pd.concat([real_sample, synth_sample], ignore_index=True)
    # Only use numeric columns for t-SNE
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    features = combined[numeric_cols]
    # t-SNE 3D
    tsne = TSNE(n_components=3, random_state=random_state, perplexity=30)
    tsne_results = tsne.fit_transform(features)
    combined['tsne-1'] = tsne_results[:,0]
    combined['tsne-2'] = tsne_results[:,1]
    combined['tsne-3'] = tsne_results[:,2]
    # 3D Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        combined['tsne-1'], combined['tsne-2'], combined['tsne-3'],
        c=combined['__source__'].map({'Real': 0, 'Synthetic': 1}),
        cmap='Set1', alpha=0.7
    )
    # Optionally, color by class/type
    if target_col and target_col in combined.columns:
        for t in combined[target_col].unique():
            idx = combined[target_col] == t
            ax.scatter(
                combined.loc[idx, 'tsne-1'],
                combined.loc[idx, 'tsne-2'],
                combined.loc[idx, 'tsne-3'],
                label=f'{t}',
                alpha=0.5
            )
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    plot_title = '3D t-SNE: Real vs Synthetic Data'
    if gan_variant:
        plot_title += f' ({gan_variant})'
    plt.title(plot_title)
    ax.legend()
    plt.show()

########## ok
########## ok
# Subplot t-SNE 3D comparison for all GAN variants
fig = plt.figure(figsize=(20, 16))
gan_variants = [
    ('VanillaGAN', df_balanced_vanillaGAN),
    ('WGAN', df_balanced_wgan),
    ('WGANGP', df_balanced_wgangp),
    ('CGAN', df_balanced_cgan)
]
from matplotlib.lines import Line2D
markers = {'Real': 'o', 'Synthetic': '^'}
for i, (title, df_bal) in enumerate(gan_variants, 1):
    ax = fig.add_subplot(2, 2, i, projection='3d')
    real_sample = df.sample(n=min(1000, len(df)), random_state=42)
    synth_sample = df_bal.sample(n=min(1000, len(df_bal)), random_state=42)
    real_sample = real_sample.copy()
    synth_sample = synth_sample.copy()
    real_sample['__source__'] = 'Real'
    synth_sample['__source__'] = 'Synthetic'
    combined = pd.concat([real_sample, synth_sample], ignore_index=True)
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    features = combined[numeric_cols]
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=3, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(features)
    combined['tsne-1'] = tsne_results[:,0]
    combined['tsne-2'] = tsne_results[:,1]
    combined['tsne-3'] = tsne_results[:,2]
    # Color by class
    classes = combined['type'].unique()
    palette = sns.color_palette('tab10', n_colors=len(classes))
    class_color = {c: palette[j] for j, c in enumerate(classes)}
    for source in ['Real', 'Synthetic']:
        for c in classes:
            idx = (combined['__source__'] == source) & (combined['type'] == c)
            ax.scatter(
                combined.loc[idx, 'tsne-1'],
                combined.loc[idx, 'tsne-2'],
                combined.loc[idx, 'tsne-3'],
                c=[class_color[c]],
                marker=markers[source],
                label=f'{source}-{c}' if i == 1 else None,  # Only label in first subplot to avoid duplicate legend
                alpha=0.7
            )
    ax.set_title(f'3D t-SNE: {title}')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    # Custom legends
    if i == 1:
        # Legend for class
        class_legend = [Line2D([0], [0], marker='o', color='w', label=str(c), markerfacecolor=class_color[c], markersize=10) for c in classes]
        # Legend for source
        source_legend = [Line2D([0], [0], marker=markers[s], color='k', label=s, linestyle='', markersize=10) for s in markers]
        ax.legend(handles=class_legend + source_legend, title='Class / Source', loc='upper right')
fig.suptitle('3D t-SNE Comparison: Real vs Synthetic Data for All GAN Variants', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

##########
# === t-SNE subplot for each GAN variant in a single figure ===
gan_variants = [
    ('VanillaGAN', df_balanced_vanillaGAN),
    ('WGAN', df_balanced_wgan),
    ('WGANGP', df_balanced_wgangp),
    ('CGAN', df_balanced_cgan)
]
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.flatten()
for i, (name, synth_df) in enumerate(gan_variants):
    plot_tsne_comparison_by_target(df, synth_df, target_col='type', ax=axes[i], title=f't-SNE: {name}')
plt.tight_layout()
plt.suptitle('t-SNE Comparison by Target for Each GAN Variant', fontsize=20, y=1.02)
plt.show()

########## time for each GAN
# df_balanced_2, timing_vanilla = oversample_with_gan(df, target_col='type', gan_type='vanilla', epochs=1000, batch_size=32)
# print(df_balanced_2['type'].value_counts())

df_balanced_vanillaGAN, timing_vanilla = oversample_with_gan(df, target_col='type', gan_type='vanilla', epochs=1000, batch_size=32)
total_time_vanilla = sum(timing_vanilla.values())
print(f"Total training time for VanillaGAN: {total_time_vanilla:.2f} seconds ({total_time_vanilla/60:.2f} min)")
print(df_balanced_vanillaGAN['type'].value_counts())

df_balanced_wgan, timing_wgan = oversample_with_gan(df, target_col='type', gan_type='wgan', epochs=1000, batch_size=32)
total_time_wgan = sum(timing_wgan.values())
print(f"Total training time for WGAN: {total_time_wgan:.2f} seconds ({total_time_wgan/60:.2f} min)")
print(df_balanced_wgan['type'].value_counts())

df_balanced_wgangp, timing_wgangp = oversample_with_gan(df, target_col='type', gan_type='wgangp', epochs=1000, batch_size=32)
total_time_wgangp = sum(timing_wgangp.values())
print(f"Total training time for WGANGP: {total_time_wgangp:.2f} seconds ({total_time_wgangp/60:.2f} min)")
print(df_balanced_wgangp['type'].value_counts())

df_balanced_cgan, timing_cgan = oversample_with_gan(df, target_col='type', gan_type='cgan', epochs=1000, batch_size=32)
total_time_cgan = sum(timing_cgan.values())
print(f"Total training time for CGAN: {total_time_cgan:.2f} seconds ({total_time_cgan/60:.2f} min)")
print(df_balanced_cgan['type'].value_counts())

# After running oversample_with_gan for all variants, save timing info as a table
all_timing = {**timing_vanilla, **timing_wgan, **timing_wgangp, **timing_cgan}
timing_df = pd.DataFrame([
    {'gan_type': k.split('_')[-1], 'class_idx': int(k.split('_')[1]), 'seconds': v}
    for k, v in all_timing.items()
])
# Ensure 'minutes' column exists before any groupby/access
if 'minutes' not in timing_df.columns:
    timing_df['minutes'] = timing_df['seconds'] / 60
print(timing_df)
timing_df.to_csv('../gan_training_timing.csv', index=False)

# Calculate and save total training time for each GAN variant
# (Integrated as requested)
total_time_per_gan = timing_df.groupby('gan_type')['seconds'].sum()
total_time_per_gan_min = timing_df.groupby('gan_type')['minutes'].sum()
total_time_per_gan_df = pd.DataFrame({'seconds': total_time_per_gan, 'minutes': total_time_per_gan_min})
print("Total training time for each GAN variant:")
# print(total_time_per_gan_df)
for gan, row in total_time_per_gan_df.iterrows():
    print(f"{gan}: {row['seconds']:.2f} seconds ({row['minutes']:.2f} min)")
total_time_per_gan_df.to_csv('../gan_total_training_time.csv')

# Save total training times for all variants to a CSV
import pandas as pd
per_variant_total_time = pd.DataFrame([
    {'gan_type': 'vanilla', 'total_seconds': total_time_vanilla, 'total_minutes': total_time_vanilla/60},
    {'gan_type': 'wgan', 'total_seconds': total_time_wgan, 'total_minutes': total_time_wgan/60},
    {'gan_type': 'wgangp', 'total_seconds': total_time_wgangp, 'total_minutes': total_time_wgangp/60},
    {'gan_type': 'cgan', 'total_seconds': total_time_cgan, 'total_minutes': total_time_cgan/60},
])

# Sort by total_seconds ascending
per_variant_total_time = per_variant_total_time.sort_values(by='total_seconds', ascending=True)
# Show as table in terminal
try:
    # from tabulate import tabulate
    print("\nTotal Training Time per GAN Variant (sorted):")
    print(tabulate(per_variant_total_time, headers='keys', tablefmt='psql', showindex=False))
except ImportError:
    print(per_variant_total_time)

# ===============================================================================
######### label to 0/1 and minmax

# Encode label column: all non-zero to 1, zero stays 0
# if 'label' in df_balanced_2.columns:
#     df_balanced_2['label'] = (df_balanced_2['label'] != 0).astype(int)
# df_balanced_2['label'].unique()

# vanillaGAN
if 'label' in df_balanced_vanillaGAN.columns:
    df_balanced_vanillaGAN['label'] = (df_balanced_vanillaGAN['label'] != 0).astype(int)
df_balanced_vanillaGAN['label'].unique()

# wgan
if 'label' in df_balanced_wgan.columns:
    df_balanced_wgan['label'] = (df_balanced_wgan['label'] != 0).astype(int)
df_balanced_wgan['label'].unique()

# wgangp
if 'label' in df_balanced_wgangp.columns:
    df_balanced_wgangp['label'] = (df_balanced_wgangp['label'] != 0).astype(int)
df_balanced_wgangp['label'].unique()

# cgan
if 'label' in df_balanced_cgan.columns:
    df_balanced_cgan['label'] = (df_balanced_cgan['label'] != 0).astype(int)
df_balanced_cgan['label'].unique()

# # Apply MinMax scaling to all columns except 'label' and 'type'
# cols_to_scale = [col for col in df_balanced_2.columns if col not in ['label', 'type']]
# if cols_to_scale:
#     scaler = MinMaxScaler()
#     df_balanced_2[cols_to_scale] = scaler.fit_transform(df_balanced_2[cols_to_scale])
    
# ===============================================================================
# ================= KS and JS Visualization for Each GAN Variant =================
# Collect KS and JS for each GAN variant
ks_dict = {}
js_dict = {}
gan_variants = {
    # 'VanillaGAN': df_balanced_2,
    'VanillaGAN': df_balanced_vanillaGAN,
    'WGAN': df_balanced_wgan,
    'WGANGP': df_balanced_wgangp,
    'CGAN': df_balanced_cgan
}
for name, synth_df in gan_variants.items():
    ks_results, js_results = evaluate_statistical_similarity(df, synth_df, target_col='type')
    ks_dict[name] = {k: v['ks_stat'] for k, v in ks_results.items()}
    js_dict[name] = js_results
ks_df = pd.DataFrame(ks_dict)
js_df = pd.DataFrame(js_dict)

# Plot KS Statistic
plt.figure(figsize=(14, 6))
ks_df.plot(kind='bar')
plt.title('KS Statistic per Feature for Each GAN Variant')
plt.ylabel('KS Statistic')
plt.xlabel('Feature')
plt.xticks(rotation=45, ha='right')
plt.legend(title='GAN Variant')
plt.tight_layout()
plt.show()

# Plot JS Divergence
plt.figure(figsize=(14, 6))
js_df.plot(kind='bar')
plt.title('JS Divergence per Feature for Each GAN Variant')
plt.ylabel('JS Divergence')
plt.xlabel('Feature')
plt.xticks(rotation=45, ha='right')
plt.legend(title='GAN Variant')
plt.tight_layout()
plt.show()

# Optionally, as heatmaps
plt.figure(figsize=(10, 8))
sns.heatmap(ks_df, annot=True, cmap='viridis')
plt.title('KS Statistic Heatmap')
plt.ylabel('Feature')
plt.xlabel('GAN Variant')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(js_df, annot=True, cmap='magma')
plt.title('JS Divergence Heatmap')
plt.ylabel('Feature')
plt.xlabel('GAN Variant')
plt.show()
# ===============================================================================

# Example usage:
# df_balanced = oversample_with_gan(data_simulator1, target_col='type', gan_type='cgan', epochs=500, batch_size=32)
# print(df_balanced['type'].value_counts())

# Example usage for reporting:
# df_balanced = oversample_with_gan(data_simulator1, target_col='type', gan_type='cgan', epochs=500, batch_size=32)
# synthetic_data_report(data_simulator1, df_balanced, target_col='type', gan_type='cgan', print_report=True)

# Example usage:
# if gan_type == 'vanilla', 'wgan', 'wgangp', 'cgan'
# 'vanilla' = input_dim, latent_dim=32, epochs=1000, batch_size=32, optimizer='adam', loss='binary_crossentropy'
# 'WGAN' = input_dim, latent_dim=32, clip_value=0.01, epochs=1000, batch_size=32, n_critic=5, optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00005), loss=wasserstein
# 'WGAN-GP' = input_dim, latent_dim=32, gp_weight=10, epochs=1000, batch_size=32, n_critic=5, optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss=wasserstein
# 'CGAN' = input_dim, n_classes, latent_dim=32, epochs=1000, batch_size=32, optimizer='adam', loss='binary_crossentropy'

# ================= oversampling for Each GAN Variant =================
# df_balanced = oversample_with_gan(df, target_col='type', gan_type='vanilla', epochs=1000, batch_size=32)
# df_balanced_vanillaGAN = oversample_with_gan(df, target_col='type', gan_type='vanilla', epochs=1000, batch_size=32)

# ===============================================================================
# Example usage for reporting:
# df_balanced_2 = oversample_with_gan(data_simulator1, target_col='type', gan_type='cgan', epochs=500, batch_size=32)

# df_balanced_vanillaGAN
synthetic_data_report(df, df_balanced_vanillaGAN, target_col='type', gan_type='vanilla', print_report=True)
evaluate_statistical_similarity(df, df_balanced_vanillaGAN, target_col='type')
evaluate_class_distribution(df, df_balanced_vanillaGAN, target_col='type')
evaluate_classifier(df, df_balanced_vanillaGAN, target_col='type', gan_variant='VanillaGAN')

# wgan
synthetic_data_report(df, df_balanced_wgan, target_col='type', gan_type='wgan', print_report=True)
evaluate_statistical_similarity(df, df_balanced_wgan, target_col='type')
evaluate_class_distribution(df, df_balanced_wgan, target_col='type')
evaluate_classifier(df, df_balanced_wgan, target_col='type', gan_variant='WGAN')

# wgangp
synthetic_data_report(df, df_balanced_wgangp, target_col='type', gan_type='wgangp', print_report=True)
evaluate_statistical_similarity(df, df_balanced_wgangp, target_col='type')
evaluate_class_distribution(df, df_balanced_wgangp, target_col='type')
evaluate_classifier(df, df_balanced_wgangp, target_col='type', gan_variant='WGANGP')

# cgan
synthetic_data_report(df, df_balanced_cgan, target_col='type', gan_type='cgan', print_report=True)
evaluate_statistical_similarity(df, df_balanced_cgan, target_col='type')
evaluate_class_distribution(df, df_balanced_cgan, target_col='type')
evaluate_classifier(df, df_balanced_cgan, target_col='type', gan_variant='CGAN')



# t-SNE comparison plot

# vanillaGAN
# plot_tsne_comparison_by_target
plot_tsne_comparison_by_target(df, df_balanced_vanillaGAN, target_col='type', gan_variant='VanillaGAN')
# def plot_tsne_comparison without legend
plot_tsne_comparison(df, df_balanced_vanillaGAN, gan_variant='VanillaGAN')
# 3D t-SNE comparison plot
plot_tsne_comparison_3d(df, df_balanced_vanillaGAN, target_col='type', gan_variant='VanillaGAN')
plot_tsne_comparison_3d(df, df_balanced_vanillaGAN, target_col='label', gan_variant='VanillaGAN')

# wgan
# plot_tsne_comparison_by_target
plot_tsne_comparison_by_target(df, df_balanced_wgan, target_col='type', gan_variant='WGAN')
# def plot_tsne_comparison without legend
plot_tsne_comparison(df, df_balanced_wgan, gan_variant='WGAN')
# 3D t-SNE comparison plot
plot_tsne_comparison_3d(df, df_balanced_wgan, target_col='type', gan_variant='WGAN')
plot_tsne_comparison_3d(df, df_balanced_wgan, target_col='label', gan_variant='WGAN')

# wgangp
# plot_tsne_comparison_by_target
plot_tsne_comparison_by_target(df, df_balanced_wgangp, target_col='type', gan_variant='WGANGP')
# def plot_tsne_comparison without legend
plot_tsne_comparison(df, df_balanced_wgangp, gan_variant='WGANGP')
# 3D t-SNE comparison plot
plot_tsne_comparison_3d(df, df_balanced_wgangp, target_col='type', gan_variant='WGANGP')
plot_tsne_comparison_3d(df, df_balanced_wgangp, target_col='label', gan_variant='WGANGP')

# cgan
# plot_tsne_comparison_by_target
plot_tsne_comparison_by_target(df, df_balanced_cgan, target_col='type', gan_variant='CGAN')
# def plot_tsne_comparison without legend
plot_tsne_comparison(df, df_balanced_cgan, gan_variant='CGAN')
# 3D t-SNE comparison plot
plot_tsne_comparison_3d(df, df_balanced_cgan, target_col='type', gan_variant='CGAN')
plot_tsne_comparison_3d(df, df_balanced_cgan, target_col='label', gan_variant='CGAN')

# print(df_balanced_2.shape)
# df_balanced_2.dtypes
# df_balanced_2['type'].unique()
# df_balanced_2['label'].unique()


##########
# Check the count of category class
# counts = df_balanced_2.value_counts('type')
# print(counts)
# let's see the distribution of our target category by bar chart
# plt.figure(figsize=(10,5))
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
# plt.xticks(rotation=45, ha='right')
# ax = sns.countplot(x = 'type', data = data1, order=data1['type'].value_counts(ascending=True).index, palette = 'Set2')
# ax = sns.countplot(x = 'type', data = df_balanced_2, order=df_balanced_2['type'].value_counts(ascending=False).index, palette = 'Set2')
# for container in ax.containers:
#     ax.bar_label(container)
# ax.set_title("ToN_IoT_resampled_balanced Count of Normal and Attack Types")
##########

# Example usage for reporting:
df_balanced = oversample_with_gan(data_simulator1, target_col='type', gan_type='vanilla', epochs=1000, batch_size=32)
synthetic_data_report(data_simulator1, df_balanced, target_col='type', gan_type='vanilla', print_report=True)

# # === t-SNE subplot for each GAN variant in a single figure ===
# gan_variants = [
#     ('VanillaGAN', df_balanced_vanillaGAN),
#     ('WGAN', df_balanced_wgan),
#     ('WGANGP', df_balanced_wgangp),
#     ('CGAN', df_balanced_cgan)
# ]
# fig, axes = plt.subplots(2, 2, figsize=(18, 14))
# axes = axes.flatten()
# for i, (name, synth_df) in enumerate(gan_variants):
#     plot_tsne_comparison_by_target(df, synth_df, target_col='type', ax=axes[i], title=f't-SNE: {name}')
# plt.tight_layout()
# plt.suptitle('t-SNE Comparison by Target for Each GAN Variant', fontsize=20, y=1.02)
# plt.show()


# === Example usage of classifiers:
# Only Random Forest
# evaluate_classifier(real_df, synth_df, target_col, classifier='randomforest')
# Only XGBoost
# evaluate_classifier(real_df, synth_df, target_col, classifier='xgboost')
# Both (default)
# evaluate_classifier(real_df, synth_df, target_col)

# evaluate_classifier(df, df_balanced_vanillaGAN, target_col='type', classifier='xgboost', gan_variant='VanillaGAN')
evaluate_classifier(df, df_balanced_vanillaGAN, target_col='type', classifier='all', gan_variant='VanillaGAN')
evaluate_classifier(df, df_balanced_wgan, target_col='type', classifier='all', gan_variant='WGAN')
evaluate_classifier(df, df_balanced_wgangp, target_col='type', classifier='all', gan_variant='WGANGP')
evaluate_classifier(df, df_balanced_cgan, target_col='type', classifier='all', gan_variant='CGAN')
evaluate_classifier(real_df, synth_df, target_col='type')



# === Accuracies Comparison Table: GAN variants accuracies by all classifiers ===
# Build metrics_table for all GAN variants and classifiers
metrics_table = []
gan_variants = {
    # 'VanillaGAN': df_balanced_2,
    'VanillaGAN': df_balanced_vanillaGAN,
    'WGAN': df_balanced_wgan,
    'WGANGP': df_balanced_wgangp,
    'CGAN': df_balanced_cgan
}
for gan_name, synth_df in gan_variants.items():
    clf_results = evaluate_classifier(df, synth_df, target_col='type', classifier='all', gan_variant=gan_name)
    row = {'GAN': gan_name}
    if 'RandomForest' in clf_results and clf_results['RandomForest'] is not None:
        row['RF_Accuracy'] = clf_results['RandomForest']['accuracy']
    if 'XGBoost' in clf_results and clf_results['XGBoost'] is not None:
        row['XGB_Accuracy'] = clf_results['XGBoost']['accuracy']
    if 'KNN' in clf_results and clf_results['KNN'] is not None:
        row['KNN_Accuracy'] = clf_results['KNN']['accuracy']
    metrics_table.append(row)
metrics_df = pd.DataFrame(metrics_table)

# Build accuracy_comparison_csv from metrics_df
accuracy_comparison_csv = metrics_df[['GAN']].copy()
if 'RF_Accuracy' in metrics_df.columns:
    accuracy_comparison_csv['RandomForest'] = metrics_df['RF_Accuracy']
if 'XGB_Accuracy' in metrics_df.columns:
    accuracy_comparison_csv['XGBoost'] = metrics_df['XGB_Accuracy']
if 'KNN_Accuracy' in metrics_df.columns:
    accuracy_comparison_csv['KNN'] = metrics_df['KNN_Accuracy']
# Sort by all numeric columns descending
numeric_cols = ['RandomForest', 'XGBoost', 'KNN']
existing_numeric_cols = [col for col in numeric_cols if col in accuracy_comparison_csv.columns]
if existing_numeric_cols:
    accuracy_comparison_csv = accuracy_comparison_csv.sort_values(by=existing_numeric_cols, ascending=[False]*len(existing_numeric_cols))

# Bar plot for accuracy_comparison
barplot_cols = ['RandomForest', 'XGBoost', 'KNN']
barplot_cols_present = [col for col in barplot_cols if col in accuracy_comparison_csv.columns]
fig_bar, axes_bar = plt.subplots(1, len(barplot_cols_present), figsize=(7*len(barplot_cols_present), 6))
if len(barplot_cols_present) == 1:
    axes_bar = [axes_bar]
for i, col in enumerate(barplot_cols_present):
    sorted_df = accuracy_comparison_csv.sort_values(by=col, ascending=False)
    ax = axes_bar[i]
    bars = ax.bar(sorted_df['GAN'], sorted_df[col], color='skyblue')
    ax.set_title(f'{col} Accuracy by GAN Variant', fontsize=16)
    ax.set_xlabel('GAN Variant')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('../gan_classifier_accuracy_barplot.png', bbox_inches='tight')
plt.show()

#########

# Bar plot for total training time per GAN variant
bar_df = per_variant_total_time.sort_values(by='total_seconds', ascending=True)
fig_time, ax_time = plt.subplots(figsize=(8, 6))
bars = ax_time.bar(bar_df['gan_type'], bar_df['total_seconds'], color='mediumseagreen')
ax_time.set_title('Total Training Time per GAN Variant', fontsize=16)
ax_time.set_xlabel('GAN Variant')
ax_time.set_ylabel('Total Training Time (seconds)')
for bar in bars:
    height = bar.get_height()
    ax_time.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('../gan_total_training_time_barplot.png', bbox_inches='tight')
plt.show()

########## time plot for each class
# Bar Plot training time for each GAN variant and class
plt.figure(figsize=(12, 6))
sns.barplot(data=timing_df, x='class_idx', y='seconds', hue='gan_type')
plt.title('Training Time per Class for Each GAN Variant')
plt.xlabel('Class Index')
plt.ylabel('Training Time (seconds)')
plt.legend(title='GAN Variant')
plt.tight_layout()
plt.show()

##########

df_balanced_cgan['label'].unique()

df_balanced_cgan.to_csv('../cgan_balanced.csv', index=False)
cgan_balanced = pd.read_csv('../cgan_balanced.csv')
# Apply MinMax scaling to all columns except 'label' and 'type'
cols_to_scale = [col for col in cgan_balanced.columns if col not in ['label', 'type']]
if cols_to_scale:
    scaler = MinMaxScaler()
    cgan_balanced[cols_to_scale] = scaler.fit_transform(cgan_balanced[cols_to_scale])
cgan_balanced.head()
cgan_balanced.to_csv('../cgan_balanced_minmax.csv', index=False)



