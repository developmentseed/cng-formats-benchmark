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

{{/*
Shared GDAL /vsis3 environment for the runner's read metric and TiTiler. An
explicit/derived endpoint (MinIO or a non-AWS provider) switches GDAL to
host:port + path-style; HTTPS is inferred from the endpoint scheme.
*/}}
{{- define "cng-benchmark.gdalEnv" -}}
{{- $endpoint := include "cng-benchmark.s3Endpoint" . }}
{{- if $endpoint }}
- name: AWS_S3_ENDPOINT
  value: {{ trimPrefix "https://" (trimPrefix "http://" $endpoint) | quote }}
- name: AWS_VIRTUAL_HOSTING
  value: "FALSE"
- name: AWS_HTTPS
  value: {{ hasPrefix "https://" $endpoint | ternary "YES" "NO" | quote }}
{{- end }}
- name: GDAL_DISABLE_READDIR_ON_OPEN
  value: EMPTY_DIR
# NB: do not set CPL_VSIL_CURL_ALLOWED_EXTENSIONS. A .tif-only allowlist blocks
# GDAL from reading delivery-archive sources over /vsis3 (S1/S2 are addressed as
# /vsizip//vsis3/<scene>.zip/...), and the benchmark also reads .nc/.laz/etc.
# GDAL_DISABLE_READDIR_ON_OPEN already prevents the sidecar-probing this would
# otherwise optimise.
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

{{/* Mount path for the source private-CA bundle. */}}
{{- define "cng-benchmark.sourceCaPath" -}}
{{- printf "/etc/cng/source-ca/%s" .Values.s3Source.caBundleKey -}}
{{- end }}

{{/* Name of the Secret holding source S3 credentials. */}}
{{- define "cng-benchmark.s3SourceSecretName" -}}
{{- if .Values.s3Source.existingSecret -}}
{{- .Values.s3Source.existingSecret -}}
{{- else -}}
{{- printf "%s-s3-source" (include "cng-benchmark.fullname" .) -}}
{{- end -}}
{{- end }}

{{/*
SOURCE_AWS_* environment for the runner, emitted only when a distinct source
provider is configured (s3Source.enabled). storage.s3_profile("source") reads
these first and falls back to the bare AWS_* (the sink), so the synthetic
single-endpoint path needs none of this.
*/}}
{{- define "cng-benchmark.sourceEnv" -}}
{{- if .Values.s3Source.enabled }}
- name: SOURCE_AWS_ACCESS_KEY_ID
  valueFrom:
    secretKeyRef:
      name: {{ include "cng-benchmark.s3SourceSecretName" . }}
      key: AWS_ACCESS_KEY_ID
- name: SOURCE_AWS_SECRET_ACCESS_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "cng-benchmark.s3SourceSecretName" . }}
      key: AWS_SECRET_ACCESS_KEY
{{- if .Values.s3Source.region }}
- name: SOURCE_AWS_DEFAULT_REGION
  value: {{ .Values.s3Source.region | quote }}
{{- end }}
{{- if .Values.s3Source.endpoint }}
- name: SOURCE_AWS_ENDPOINT_URL
  value: {{ .Values.s3Source.endpoint | quote }}
- name: SOURCE_AWS_ENDPOINT_URL_S3
  value: {{ .Values.s3Source.endpoint | quote }}
{{- end }}
{{- if .Values.s3Source.caBundleSecret }}
- name: SOURCE_AWS_CA_BUNDLE
  value: {{ include "cng-benchmark.sourceCaPath" . | quote }}
{{- end }}
{{- end }}
{{- end }}
