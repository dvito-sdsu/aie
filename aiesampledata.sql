USE aie;

-- USERS
INSERT INTO Users (username, email, password_hash, role, longitude, latitude) VALUES
('farm_ali', 'farm_ali@example.com', 'hash_ali_001', 'farmer', -96.797000, 32.776700),
('research_ben', 'research_ben@example.com', 'hash_ben_002', 'researcher', -96.800000, 32.780000),
('admin_cara', 'admin_cara@example.com', 'hash_cara_003', 'admin', -96.810000, 32.770000),
('farm_diego', 'farm_diego@example.com', 'hash_diego_004', 'farmer', -96.820000, 32.760000),
('research_ella', 'research_ella@example.com', 'hash_ella_005', 'researcher', -96.830000, 32.750000),
('farm_faisal', 'farm_faisal@example.com', 'hash_faisal_006', 'farmer', -96.840000, 32.740000),
('admin_gina', 'admin_gina@example.com', 'hash_gina_007', 'admin', -96.850000, 32.730000),
('research_hugo', 'research_hugo@example.com', 'hash_hugo_008', 'researcher', -96.860000, 32.720000),
('farm_ivy', 'farm_ivy@example.com', 'hash_ivy_009', 'farmer', -96.870000, 32.710000),
('research_jules', 'research_jules@example.com', 'hash_jules_010', 'researcher', -96.880000, 32.700000);

-- DISEASES
INSERT INTO Diseases (name, symptoms, crop_type, severity, transmissibility) VALUES
('Early Blight', 'Brown concentric leaf spots, yellowing, defoliation', 'Tomato', 'medium', 0.65000),
('Late Blight', 'Dark lesions, rapid rot, white fungal growth', 'Potato', 'high', 0.85000),
('Powdery Mildew', 'White powdery coating on leaves and stems', 'Wheat', 'medium', 0.55000),
('Rust', 'Orange-brown pustules on leaves', 'Corn', 'medium', 0.60000),
('Leaf Spot', 'Circular necrotic spots with yellow halos', 'Soybean', 'low', 0.30000),
('Bacterial Wilt', 'Sudden wilting, stem browning, plant collapse', 'Pepper', 'high', 0.78000),
('Anthracnose', 'Dark sunken lesions on fruit and stems', 'Mango', 'medium', 0.50000),
('Downy Mildew', 'Yellow patches, gray growth underneath leaves', 'Cucumber', 'high', 0.72000),
('Fusarium Wilt', 'Yellowing, vascular browning, stunting', 'Banana', 'high', 0.81000),
('Root Rot', 'Wilting, root decay, poor vigor', 'Cotton', 'medium', 0.45000);

-- TREATMENTS
INSERT INTO Treatment (disease_id, description, frequency, source) VALUES
(1, 'Apply copper-based fungicide and remove infected leaves.', 'Every 7 days', 'Agri Extension Guide'),
(2, 'Use resistant varieties and fungicide at first sign.', 'Every 5 days', 'Crop Disease Handbook'),
(3, 'Apply sulfur spray and improve airflow.', 'Every 10 days', 'University Research Note'),
(4, 'Rotate crops and apply approved fungicide.', 'Every 14 days', 'State Agriculture Bulletin'),
(5, 'Use biofungicide and sanitize tools between fields.', 'Weekly', 'Plant Health Manual'),
(6, 'Remove affected plants and improve drainage.', 'As needed', 'Extension Service'),
(7, 'Prune infected tissue and apply preventive fungicide.', 'Every 7 days', 'Tropical Crop Advisory'),
(8, 'Reduce leaf wetness and apply fungicide.', 'Every 5 days', 'Greenhouse Disease Guide'),
(9, 'Use disease-free planting material and soil treatment.', 'One-time plus monitoring', 'Banana Health Report'),
(10, 'Improve soil drainage and apply root pathogen control.', 'Every 10 days', 'Soil Health Handbook');

-- MODEL METADATA
INSERT INTO ModelMetadata (version, framework, trained_date, accuracy) VALUES
('v1.0', 'TensorFlow', '2025-01-15', 0.9123),
('v1.1', 'TensorFlow', '2025-03-10', 0.9234),
('v1.2', 'PyTorch', '2025-05-22', 0.9311),
('v2.0', 'PyTorch', '2025-08-14', 0.9445),
('v2.1', 'PyTorch', '2025-10-09', 0.9510),
('v2.2', 'TensorFlow', '2025-11-20', 0.9587),
('v3.0', 'PyTorch', '2026-01-12', 0.9622),
('v3.1', 'TensorFlow', '2026-02-18', 0.9650),
('v3.2', 'PyTorch', '2026-03-25', 0.9688),
('v4.0', 'TensorFlow', '2026-04-30', 0.9725);

-- IMAGES
INSERT INTO Images (user_id, file_path, status) VALUES
(1, 'https://blob.core.windows.net/aie/images/img_001.jpg', 'processed'),
(2, 'https://blob.core.windows.net/aie/images/img_002.jpg', 'processed'),
(4, 'https://blob.core.windows.net/aie/images/img_003.jpg', 'pending'),
(5, 'https://blob.core.windows.net/aie/images/img_004.jpg', 'processed'),
(6, 'https://blob.core.windows.net/aie/images/img_005.jpg', 'error'),
(7, 'https://blob.core.windows.net/aie/images/img_006.jpg', 'processed'),
(8, 'https://blob.core.windows.net/aie/images/img_007.jpg', 'pending'),
(9, 'https://blob.core.windows.net/aie/images/img_008.jpg', 'processed'),
(10, 'https://blob.core.windows.net/aie/images/img_009.jpg', 'processed'),
(3, 'https://blob.core.windows.net/aie/images/img_010.jpg', 'pending');

-- DETECTION RESULTS
INSERT INTO DetectionResults (image_id, disease_id, model_id, confidence, bounding_box) VALUES
(1, 1, 1, 0.9234, '{"x":120,"y":80,"width":210,"height":160}'),
(2, 2, 2, 0.9456, '{"x":95,"y":60,"width":180,"height":140}'),
(3, 3, 3, 0.8812, '{"x":30,"y":40,"width":250,"height":190}'),
(4, 4, 4, 0.9021, '{"x":70,"y":100,"width":220,"height":170}'),
(5, 5, 5, 0.7905, '{"x":45,"y":55,"width":160,"height":130}'),
(6, 6, 6, 0.9677, '{"x":110,"y":75,"width":200,"height":150}'),
(7, 7, 7, 0.8533, '{"x":140,"y":90,"width":240,"height":180}'),
(8, 8, 8, 0.9188, '{"x":60,"y":35,"width":175,"height":125}'),
(9, 9, 9, 0.9364, '{"x":100,"y":65,"width":190,"height":145}'),
(10, 10, 10, 0.8749, '{"x":20,"y":25,"width":155,"height":115}');

-- OUTBREAK
INSERT INTO Outbreak (disease_id, longitude, latitude, radius, amount, start_date, end_date) VALUES
(1, -96.797000, 32.776700, 0.05000, 2, '2026-04-01', '2027-04-01'),
(2, -96.800000, 32.780000, 0.08000, 4, '2026-04-03', '2027-04-03'),
(3, -96.810000, 32.770000, 0.04000, 6, '2026-04-04', '2027-04-04'),
(4, -96.820000, 32.760000, 0.06000, 8, '2026-04-05', '2027-04-05'),
(5, -96.830000, 32.750000, 0.03000, 10, '2026-04-06', '2027-04-06'),
(6, -96.840000, 32.740000, 0.07000, 12, '2026-04-07', '2027-04-07'),
(7, -96.850000, 32.730000, 0.04500, 14, '2026-04-08', '2027-04-08'),
(8, -96.860000, 32.720000, 0.05500, 16, '2026-04-09', '2027-04-09'),
(9, -96.870000, 32.710000, 0.06500, 18, '2026-04-10', '2027-04-10'),
(10, -96.880000, 32.700000, 0.03500, 20, '2026-04-11', '2027-04-11');

-- TREATMENT LOGS
INSERT INTO TreatmentLogs (user_id, image_id, disease_id, treatment_id, outcome, notes) VALUES
(1, 1, 1, 1, 'Resolved', 'Symptoms reduced after first spray cycle.'),
(2, 2, 2, 2, 'No Improvements', 'Disease spread before treatment could start.'),
(4, 3, 3, 3, 'Resolved', 'Improved airflow and sulfur spray helped.'),
(5, 4, 4, 4, 'No Improvements', 'Wet weather limited effectiveness.'),
(6, 5, 5, 5, 'Resolved', 'Biofungicide slowed lesion growth.'),
(7, 6, 6, 6, 'Resolved', 'Drainage correction stabilized plants.'),
(8, 7, 7, 7, 'No Improvements', 'Advanced infection observed at treatment time.'),
(9, 8, 8, 8, 'Resolved', 'Leaf wetness reduced significantly.'),
(10, 9, 9, 9, 'No Improvements', 'Soil infection was already severe.'),
(3, 10, 10, 10, 'Resolved', 'Root health improved after treatment.');

