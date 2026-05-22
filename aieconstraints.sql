
USE aie;

ALTER TABLE TreatmentLogs
ADD CONSTRAINT FK_TreatmentLogs_Users foreign key (user_id) references users(user_id);

ALTER TABLE TreatmentLogs
ADD CONSTRAINT FK_TreatmentLogs_Images foreign key (image_id) references Images(image_id);

ALTER TABLE TreatmentLogs
ADD CONSTRAINT FK_TreatmentLogs_Diseases foreign key (disease_id) references Diseases(disease_id);

ALTER TABLE TreatmentLogs
ADD CONSTRAINT FK_TreatmentLogs_Treatment foreign key (treatment_id) references Treatment(treatment_id);

ALTER TABLE images
add constraint FK_Images_Users foreign key (user_id) references Users(user_id);

alter table detectionresults
add constraint FK_DetectionResults_Images foreign key (image_id) references Images(image_id);

alter table detectionresults
add constraint FK_DetectionResults_Diseases foreign key (disease_id) references Diseases(disease_id);

alter table detectionresults
add constraint FK_DetectionResults_ModelMetaData foreign key (model_id) references ModelMetaData(model_id);

alter table Treatment
add constraint FK_Treatment_Diseases foreign key (disease_id) references Diseases(disease_id);

alter table Outbreak
add constraint FK_Outbreak_Diseases foreign key (disease_id) references Diseases(disease_id);
