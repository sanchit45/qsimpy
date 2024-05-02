// Benchmark was created by MQT Bench on 2023-06-29
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: v1.0.0
// Qiskit version: {'qiskit-terra': '0.24.1', 'qiskit-aer': '0.12.0', 'qiskit-ignis': None, 'qiskit-ibmq-provider': '0.20.2', 'qiskit': '0.43.1', 'qiskit-nature': '0.6.2', 'qiskit-finance': '0.3.4', 'qiskit-optimization': '0.5.0', 'qiskit-machine-learning': '0.6.1'}

OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg meas[4];
u2(2.0,-pi) q[0];
u2(2.0,-pi) q[1];
cx q[0],q[1];
p(9.17283818781952) q[1];
cx q[0],q[1];
u2(2.0,-pi) q[2];
cx q[0],q[2];
p(9.17283818781952) q[2];
cx q[0],q[2];
cx q[1],q[2];
p(9.17283818781952) q[2];
cx q[1],q[2];
u2(2.0,-pi) q[3];
cx q[0],q[3];
p(9.17283818781952) q[3];
cx q[0],q[3];
u2(2.0,-pi) q[0];
cx q[1],q[3];
p(9.17283818781952) q[3];
cx q[1],q[3];
u2(2.0,-pi) q[1];
cx q[0],q[1];
p(9.17283818781952) q[1];
cx q[0],q[1];
cx q[2],q[3];
p(9.17283818781952) q[3];
cx q[2],q[3];
u2(2.0,-pi) q[2];
cx q[0],q[2];
p(9.17283818781952) q[2];
cx q[0],q[2];
cx q[1],q[2];
p(9.17283818781952) q[2];
cx q[1],q[2];
u2(2.0,-pi) q[3];
cx q[0],q[3];
p(9.17283818781952) q[3];
cx q[0],q[3];
ry(0.514130872786957) q[0];
cx q[1],q[3];
p(9.17283818781952) q[3];
cx q[1],q[3];
ry(0.785197377310323) q[1];
cx q[2],q[3];
p(9.17283818781952) q[3];
cx q[2],q[3];
ry(0.23845470847672) q[2];
ry(0.843907811825603) q[3];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
ry(0.847604038975508) q[0];
ry(0.649598624433769) q[1];
ry(0.117021243132337) q[2];
ry(0.489351165434324) q[3];
barrier q[0],q[1],q[2],q[3];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];