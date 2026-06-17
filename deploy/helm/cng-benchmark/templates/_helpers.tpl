{{/* Expand the name of the chart. */}}
{{- define "cng-benchmark.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/* Fully qualified app name. */}}
{{- define "cng-benchmark.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{- define "cng-benchmark.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
app.kubernetes.io/name: {{ include "cng-benchmark.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/* Name of the Secret holding S3 credentials. */}}
{{- define "cng-benchmark.s3SecretName" -}}
{{- if .Values.s3.existingSecret -}}
{{- .Values.s3.existingSecret -}}
{{- else -}}
{{- printf "%s-s3" (include "cng-benchmark.fullname" .) -}}
{{- end -}}
{{- end }}

{{/* Name of the in-cluster MinIO Service. */}}
{{- define "cng-benchmark.minioName" -}}
{{- printf "%s-minio" (include "cng-benchmark.fullname" .) -}}
{{- end }}

{{/*
Effective S3 endpoint URL. An explicit s3.endpoint wins; otherwise, if the
in-cluster MinIO stand-in is enabled, point at its Service; otherwise empty
(use the provider's default endpoint, i.e. real AWS/Scaleway via region).
*/}}
{{- define "cng-benchmark.s3Endpoint" -}}
{{- if .Values.s3.endpoint -}}
{{- .Values.s3.endpoint -}}
{{- else if .Values.minio.enabled -}}
{{- printf "http://%s:9000" (include "cng-benchmark.minioName" .) -}}
{{- end -}}
{{- end }}

{{/* Shared boto3/AWS environment for the runner. */}}
{{- define "cng-benchmark.s3Env" -}}
- name: AWS_ACCESS_KEY_ID
  valueFrom:
    secretKeyRef:
      name: {{ include "cng-benchmark.s3SecretName" . }}
      key: AWS_ACCESS_KEY_ID
- name: AWS_SECRET_ACCESS_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "cng-benchmark.s3SecretName" . }}
      key: AWS_SECRET_ACCESS_KEY
- name: AWS_DEFAULT_REGION
  value: {{ .Values.s3.region | quote }}
{{- $endpoint := include "cng-benchmark.s3Endpoint" . }}
{{- if $endpoint }}
- name: AWS_ENDPOINT_URL
  value: {{ $endpoint | quote }}
- name: AWS_ENDPOINT_URL_S3
  value: {{ $endpoint | quote }}
{{- end }}
{{- end }}
