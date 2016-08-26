inline float4 transform4f(const float16 m,float4 v){
	return (float4) (
			dot(m.s048c, v),
			dot(m.s159d, v),
			dot(m.s26ae, v),
			dot(m.s37bf, v)
		);
}
kernel void TransformV4f(global float4* vector,float16 pose){
	uint id=get_global_id(0);
	vector[id]=transform4f(pose,vector[id]);
}
kernel void test_kernel(global float4* vector,float16 pose){
	uint id=get_global_id(0);
	vector[id]=transform4f(pose,vector[id]);
}
